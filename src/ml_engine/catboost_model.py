"""
catboost_model.py — CatBoost-based predictor for 3-way match outcomes
(Home Win / Draw / Away Win).

Features are built from FIFA rankings, Dixon-Coles attack/defence
parameters, and venue/competition-type indicators.  Training uses a
chronological 80/20 split with early stopping.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result encoding
# ---------------------------------------------------------------------------
_RESULT_HOME = 0
_RESULT_DRAW = 1
_RESULT_AWAY = 2


def _encode_result(home_score: int, away_score: int) -> int:
    if home_score > away_score:
        return _RESULT_HOME
    if home_score == away_score:
        return _RESULT_DRAW
    return _RESULT_AWAY


# ---------------------------------------------------------------------------
# CatBoostPredictor
# ---------------------------------------------------------------------------

class CatBoostPredictor:
    """Gradient-boosted 3-class match predictor powered by CatBoost."""

    # ---- construction ---------------------------------------------------- #

    def __init__(self, settings: dict) -> None:
        ml_cfg: dict = settings.get("ml", {})

        self._train_history_years: int = ml_cfg.get("train_history_years", 8)
        self._validation_split: float = ml_cfg.get("validation_split", 0.2)

        self._model = CatBoostClassifier(
            iterations=ml_cfg.get("catboost_iterations", 500),
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            eval_metric="MultiClass",
            early_stopping_rounds=ml_cfg.get("early_stopping_rounds", 20),
            verbose=50,
            random_seed=42,
            auto_class_weights="Balanced",
        )

        self._is_fitted: bool = False
        logger.info(
            "CatBoostPredictor created  train_years=%d  val_split=%.2f",
            self._train_history_years,
            self._validation_split,
        )

    # ---- feature engineering --------------------------------------------- #

    def _build_features(
        self,
        matches: pd.DataFrame,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
        macro_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame aligned 1-to-1 with *matches*.

        Expected columns in *matches*:
            home_team, away_team, home_score, away_score,
            tournament, neutral (bool), date (datetime-like)

        *rankings*: ``{team_name: rank}``
        *dc_params*:  ``{team_name: {"attack": float, "defence": float}}``
        *macro_df*: DataFrame with team, gdp_per_capita, population, avg_temp_celsius
        """
        dc_params = dc_params or {}

        macro_dict = {}
        if macro_df is not None and not macro_df.empty:
            for _, r in macro_df.iterrows():
                macro_dict[str(r["team"])] = {
                    "gdp_per_capita": float(r["gdp_per_capita"]),
                    "population": float(r["population"]),
                    "avg_temp_celsius": float(r["avg_temp_celsius"]),
                }

        rows: list[dict[str, Any]] = []
        for _, m in matches.iterrows():
            home = str(m["home_team"])
            away = str(m["away_team"])

            home_dc = dc_params.get(home, {})
            away_dc = dc_params.get(away, {})

            home_rank = rankings.get(home, 100)
            away_rank = rankings.get(away, 100)

            # Determine neutrality
            is_neutral_val: int
            if "neutral" in m.index and pd.notna(m.get("neutral")):
                is_neutral_val = int(bool(m["neutral"]))
            else:
                is_neutral_val = 1  # default to neutral (tournament matches)

            # Determine competitiveness
            tournament = str(m.get("tournament", "")).strip()
            is_competitive = 0 if tournament.lower() == "friendly" else 1

            home_macro = macro_dict.get(home, {"gdp_per_capita": 10000.0, "population": 1e7, "avg_temp_celsius": 15.0})
            away_macro = macro_dict.get(away, {"gdp_per_capita": 10000.0, "population": 1e7, "avg_temp_celsius": 15.0})

            rows.append(
                {
                    "home_rank": home_rank,
                    "away_rank": away_rank,
                    "rank_diff": away_rank - home_rank,
                    "home_attack": home_dc.get("attack", 0.0),
                    "home_defence": home_dc.get("defence", 0.0),
                    "away_attack": away_dc.get("attack", 0.0),
                    "away_defence": away_dc.get("defence", 0.0),
                    "is_neutral": is_neutral_val,
                    "is_competitive": is_competitive,
                    "home_gdp": home_macro["gdp_per_capita"],
                    "away_gdp": away_macro["gdp_per_capita"],
                    "home_pop": home_macro["population"],
                    "away_pop": away_macro["population"],
                    "climate_diff": home_macro["avg_temp_celsius"] - away_macro["avg_temp_celsius"],
                }
            )

        features = pd.DataFrame(rows)

        # Target: derive from scores
        if "home_score" in matches.columns and "away_score" in matches.columns:
            features["result"] = [
                _encode_result(int(r["home_score"]), int(r["away_score"]))
                for _, r in matches.iterrows()
            ]

        return features

    # ---- training -------------------------------------------------------- #

    def fit(
        self,
        matches: pd.DataFrame,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
        macro_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Train CatBoost on recent *matches*.

        Returns a dict of training metrics:
          best_iteration, validation_loss, accuracy, n_train, n_val
        """
        # ---- filter to recent history ----
        matches = matches.copy()
        if "date" in matches.columns:
            matches["date"] = pd.to_datetime(matches["date"], utc=True)
            cutoff = datetime.now(tz=timezone.utc) - pd.DateOffset(
                years=self._train_history_years
            )
            matches = matches[matches["date"] >= cutoff]
            matches = matches.sort_values("date").reset_index(drop=True)
        else:
            logger.warning("No 'date' column — using all %d matches", len(matches))

        if len(matches) < 50:
            raise ValueError(
                f"Not enough matches for training: {len(matches)}.  "
                "Need at least 50 after filtering by history window."
            )

        # ---- build features ----
        features = self._build_features(matches, rankings, dc_params, macro_df)
        y = features.pop("result")
        X = features

        # ---- chronological split (NOT random) ----
        split_idx = int(len(X) * (1 - self._validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(
            "Training CatBoost  n_train=%d  n_val=%d  features=%s",
            len(X_train),
            len(X_val),
            list(X.columns),
        )

        train_pool = Pool(X_train, label=y_train)
        val_pool = Pool(X_val, label=y_val)

        self._model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        self._is_fitted = True

        # ---- metrics ----
        best_iter = self._model.get_best_iteration()
        val_preds = self._model.predict(X_val)
        accuracy = float(np.mean(val_preds.flatten() == y_val.values))

        # CatBoost MultiClass loss is stored in eval results
        evals = self._model.get_evals_result()
        val_loss_series = evals.get("validation", {}).get("MultiClass", [])
        val_loss = float(val_loss_series[-1]) if val_loss_series else float("nan")

        metrics: dict[str, Any] = {
            "best_iteration": best_iter,
            "validation_loss": val_loss,
            "accuracy": accuracy,
            "n_train": len(X_train),
            "n_val": len(X_val),
        }
        logger.info("CatBoost training complete: %s", metrics)
        return metrics

    # ---- prediction ------------------------------------------------------ #

    def predict(
        self,
        home_team: str,
        away_team: str,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
        macro_df: pd.DataFrame | None = None,
        neutral: bool = True,
    ) -> tuple[float, float, float]:
        """
        Predict ``(p_home_win, p_draw, p_away_win)`` for a single fixture.

        Raises RuntimeError if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("CatBoostPredictor has not been fitted yet.  Call fit() first.")

        dc_params = dc_params or {}
        home_dc = dc_params.get(home_team, {})
        away_dc = dc_params.get(away_team, {})

        home_rank = rankings.get(home_team, 100)
        away_rank = rankings.get(away_team, 100)

        macro_dict = {}
        if macro_df is not None and not macro_df.empty:
            for _, r in macro_df.iterrows():
                macro_dict[str(r["team"])] = {
                    "gdp_per_capita": float(r["gdp_per_capita"]),
                    "population": float(r["population"]),
                    "avg_temp_celsius": float(r["avg_temp_celsius"]),
                }

        home_macro = macro_dict.get(home_team, {"gdp_per_capita": 10000.0, "population": 1e7, "avg_temp_celsius": 15.0})
        away_macro = macro_dict.get(away_team, {"gdp_per_capita": 10000.0, "population": 1e7, "avg_temp_celsius": 15.0})

        row = pd.DataFrame(
            [
                {
                    "home_rank": home_rank,
                    "away_rank": away_rank,
                    "rank_diff": away_rank - home_rank,
                    "home_attack": home_dc.get("attack", 0.0),
                    "home_defence": home_dc.get("defence", 0.0),
                    "away_attack": away_dc.get("attack", 0.0),
                    "away_defence": away_dc.get("defence", 0.0),
                    "is_neutral": int(neutral),
                    "is_competitive": 1,  # tournament match
                    "home_gdp": home_macro["gdp_per_capita"],
                    "away_gdp": away_macro["gdp_per_capita"],
                    "home_pop": home_macro["population"],
                    "away_pop": away_macro["population"],
                    "climate_diff": home_macro["avg_temp_celsius"] - away_macro["avg_temp_celsius"],
                }
            ]
        )

        proba = self._model.predict_proba(row)[0]
        # CatBoost returns array in class order [0, 1, 2] = [Home, Draw, Away]
        p_home = float(proba[0])
        p_draw = float(proba[1])
        p_away = float(proba[2])

        logger.debug(
            "Predict %s vs %s (neutral=%s): H=%.3f D=%.3f A=%.3f",
            home_team, away_team, neutral, p_home, p_draw, p_away,
        )
        return (p_home, p_draw, p_away)
