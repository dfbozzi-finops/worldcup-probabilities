"""
bayesian_model.py — Logistic-Regression predictor for 3-way match outcomes
(Home Win / Draw / Away Win).

Uses L2 regularisation (equivalent to a Gaussian prior → MAP estimate) as a
lightweight Bayesian baseline.  Features are standardised with
``StandardScaler`` before fitting; the scaler is persisted for inference.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result encoding (same convention as catboost_model)
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
# Feature columns — single source of truth for column ordering
# ---------------------------------------------------------------------------
_FEATURE_COLS: list[str] = [
    "home_rank",
    "away_rank",
    "rank_diff",
    "home_attack",
    "home_defence",
    "away_attack",
    "away_defence",
    "is_neutral",
    "is_competitive",
]


# ---------------------------------------------------------------------------
# BayesianPredictor
# ---------------------------------------------------------------------------

class BayesianPredictor:
    """L2-regularised multinomial logistic regression (MAP estimate)."""

    # ---- construction ---------------------------------------------------- #

    def __init__(self, settings: dict) -> None:
        ml_cfg: dict = settings.get("ml", {})

        self._train_history_years: int = ml_cfg.get("train_history_years", 8)
        self._validation_split: float = ml_cfg.get("validation_split", 0.2)

        self._model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=500,
            random_state=42,
        )

        self.scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False

        logger.info(
            "BayesianPredictor created  C=%.2f  train_years=%d",
            self._model.C,
            self._train_history_years,
        )

    # ---- feature engineering --------------------------------------------- #

    def _build_features(
        self,
        matches: pd.DataFrame,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame aligned 1-to-1 with *matches*.

        Expected columns in *matches*:
            home_team, away_team, home_score, away_score,
            tournament, neutral (bool), date (datetime-like)

        *rankings*: ``{team_name: rank}``
        *dc_params*:  ``{team_name: {"attack": float, "defence": float}}``
        """
        dc_params = dc_params or {}

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
                is_neutral_val = 1  # default to neutral

            # Determine competitiveness
            tournament = str(m.get("tournament", "")).strip()
            is_competitive = 0 if tournament.lower() == "friendly" else 1

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
                }
            )

        features = pd.DataFrame(rows, columns=_FEATURE_COLS)

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
    ) -> dict[str, Any]:
        """
        Train Logistic Regression on recent *matches*.

        Returns a dict of training metrics:
          accuracy, log_loss, n_train, n_val, coefficients_shape
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
        features = self._build_features(matches, rankings, dc_params)
        y = features.pop("result").values
        X = features.values

        # ---- chronological split (NOT random) ----
        split_idx = int(len(X) * (1 - self._validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # ---- scale features (fit on train only) ----
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        logger.info(
            "Training BayesianPredictor  n_train=%d  n_val=%d  features=%d",
            len(X_train),
            len(X_val),
            X_train.shape[1],
        )

        # ---- fit ----
        self._model.fit(X_train_scaled, y_train)
        self._is_fitted = True

        # ---- metrics ----
        val_preds = self._model.predict(X_val_scaled)
        accuracy = float(np.mean(val_preds == y_val))

        # Log-loss on validation set
        val_proba = self._model.predict_proba(X_val_scaled)
        from sklearn.metrics import log_loss as sklearn_log_loss

        logloss = float(sklearn_log_loss(y_val, val_proba, labels=[0, 1, 2]))

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "log_loss": logloss,
            "n_train": len(X_train),
            "n_val": len(X_val),
            "coefficients_shape": list(self._model.coef_.shape),
        }
        logger.info("BayesianPredictor training complete: %s", metrics)
        return metrics

    # ---- prediction ------------------------------------------------------ #

    def predict(
        self,
        home_team: str,
        away_team: str,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
        neutral: bool = True,
    ) -> tuple[float, float, float]:
        """
        Predict ``(p_home_win, p_draw, p_away_win)`` for a single fixture.

        Raises RuntimeError if the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "BayesianPredictor has not been fitted yet.  Call fit() first."
            )

        dc_params = dc_params or {}
        home_dc = dc_params.get(home_team, {})
        away_dc = dc_params.get(away_team, {})

        home_rank = rankings.get(home_team, 100)
        away_rank = rankings.get(away_team, 100)

        row = np.array(
            [[
                home_rank,
                away_rank,
                away_rank - home_rank,
                home_dc.get("attack", 0.0),
                home_dc.get("defence", 0.0),
                away_dc.get("attack", 0.0),
                away_dc.get("defence", 0.0),
                int(neutral),
                1,  # is_competitive — tournament match
            ]],
            dtype=np.float64,
        )

        row_scaled = self.scaler.transform(row)
        proba = self._model.predict_proba(row_scaled)[0]

        # sklearn returns classes in sorted order [0, 1, 2] = [Home, Draw, Away]
        p_home = float(proba[0])
        p_draw = float(proba[1])
        p_away = float(proba[2])

        logger.debug(
            "Predict %s vs %s (neutral=%s): H=%.3f D=%.3f A=%.3f",
            home_team, away_team, neutral, p_home, p_draw, p_away,
        )
        return (p_home, p_draw, p_away)
