"""
mcmc_model.py — Custom Metropolis-Hastings MCMC sampler for 3-way match
outcome prediction (Home Win / Draw / Away Win).

Implements multinomial logistic regression (softmax) with Gaussian priors
using a random-walk Metropolis-Hastings algorithm.  Provides full posterior
uncertainty quantification via credible intervals over the posterior
predictive distribution.

Dependencies: numpy, scipy, pandas, sklearn (StandardScaler only).
Does NOT require PyMC, PyTensor, or any heavy Bayesian library.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result encoding (same convention as bayesian_model / catboost_model)
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
# Feature columns — single source of truth, matches bayesian_model.py
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
# MCMCPredictor
# ---------------------------------------------------------------------------

class MCMCPredictor:
    """
    Multinomial logistic regression with full Bayesian inference via
    Metropolis-Hastings MCMC.

    The model uses the Draw outcome as the reference category:
        eta_home = X @ beta     (log-odds of Home vs Draw)
        eta_away = X @ gamma    (log-odds of Away vs Draw)

    Priors: Normal(0, 1) on all beta and gamma coefficients.

    The sampler runs multiple independent chains with adaptive proposal
    scaling during warmup to target an optimal ≈23% acceptance rate.
    """

    # ---- construction ---------------------------------------------------- #

    def __init__(self, settings: dict) -> None:
        ml_cfg: dict = settings.get("ml", {})

        self._train_history_years: int = ml_cfg.get("train_history_years", 8)
        self._validation_split: float = ml_cfg.get("validation_split", 0.2)

        # MCMC hyper-parameters
        mcmc_cfg: dict = ml_cfg.get("mcmc", {})
        self._n_warmup: int = mcmc_cfg.get("n_warmup", 500)
        self._n_draws: int = mcmc_cfg.get("n_draws", 1000)
        self._n_chains: int = mcmc_cfg.get("n_chains", 2)
        self._thin: int = mcmc_cfg.get("thin", 2)

        self.scaler: StandardScaler = StandardScaler()
        self._is_fitted: bool = False
        self._posterior_samples: np.ndarray | None = None  # (n_total, 2*n_features)
        self._n_features: int = len(_FEATURE_COLS)

        logger.info(
            "MCMCPredictor created  warmup=%d  draws=%d  chains=%d  thin=%d  "
            "train_years=%d",
            self._n_warmup,
            self._n_draws,
            self._n_chains,
            self._thin,
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

    # ---- log-posterior ---------------------------------------------------- #

    def _log_posterior(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute un-normalised log-posterior for the multinomial logistic model.

        Parameters
        ----------
        params : ndarray, shape (2 * n_features,)
            Concatenation of beta (home-vs-draw coefficients) and gamma
            (away-vs-draw coefficients).
        X : ndarray, shape (n_obs, n_features)
            Scaled feature matrix.
        y : ndarray, shape (n_obs,)
            Outcome labels — 0 = Home, 1 = Draw, 2 = Away.

        Returns
        -------
        float
            log p(params | data) ∝ log p(data | params) + log p(params).
        """
        n_feat = X.shape[1]
        beta = params[:n_feat]   # home vs draw
        gamma = params[n_feat:]  # away vs draw

        # Linear predictors — vectorised over observations
        eta_home = X @ beta   # (n_obs,)
        eta_away = X @ gamma  # (n_obs,)

        # Softmax probabilities (draw is reference category)
        # Use log-sum-exp trick for numerical stability
        max_eta = np.maximum(np.maximum(eta_home, eta_away), 0.0)
        log_denom = max_eta + np.log(
            np.exp(-max_eta) + np.exp(eta_home - max_eta) + np.exp(eta_away - max_eta)
        )

        # Log-probabilities for each category
        log_p_home = eta_home - log_denom  # (n_obs,)
        log_p_draw = -log_denom            # log(1) - log_denom = -log_denom ... but we need 0 - log_denom
        log_p_away = eta_away - log_denom  # (n_obs,)

        # Correct log_p_draw: log(exp(0)) - log_denom = 0 - log_denom
        # Already handled above since log(1) = 0.

        # Select log-probability of the observed outcome for each observation
        log_probs = np.where(
            y == _RESULT_HOME,
            log_p_home,
            np.where(y == _RESULT_AWAY, log_p_away, log_p_draw),
        )

        # Clip for safety (avoid -inf from degenerate probabilities)
        log_probs = np.clip(log_probs, np.log(1e-10), 0.0)

        # Log-likelihood
        log_lik = np.sum(log_probs)

        # Log-prior: Normal(0, 1) on each coefficient
        # log p(theta) = -0.5 * sum(theta^2)  (up to additive constant)
        log_prior = -0.5 * np.sum(params ** 2)

        return log_lik + log_prior

    # ---- single MH chain ------------------------------------------------- #

    def _run_chain(self, X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
        """
        Run one Metropolis-Hastings chain with adaptive proposal scaling.

        Parameters
        ----------
        X : ndarray, shape (n_obs, n_features)
        y : ndarray, shape (n_obs,)
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        ndarray, shape (n_kept, n_params)
            Posterior samples after discarding warmup and applying thinning.
        """
        rng = np.random.default_rng(seed)

        n_params = 2 * X.shape[1]
        total_steps = self._n_warmup + self._n_draws

        # Initialise at the origin (maximum-prior point)
        params = np.zeros(n_params, dtype=np.float64)
        log_post = self._log_posterior(params, X, y)

        # Adaptive proposal scale
        scale = 0.1
        target_accept = 0.23

        # Storage for post-warmup samples
        samples = np.empty((self._n_draws, n_params), dtype=np.float64)

        # Counters for adaptation (running window)
        adapt_window = 50
        window_accepts = 0
        total_accepts = 0

        for step in range(total_steps):
            # Propose new parameters
            proposal = params + scale * rng.standard_normal(n_params)
            log_post_new = self._log_posterior(proposal, X, y)

            # Acceptance criterion (log scale)
            log_alpha = log_post_new - log_post

            if np.log(rng.uniform()) < log_alpha:
                params = proposal
                log_post = log_post_new
                if step < self._n_warmup:
                    window_accepts += 1
                else:
                    total_accepts += 1

            # Adapt proposal scale during warmup
            if step < self._n_warmup and (step + 1) % adapt_window == 0:
                accept_rate = window_accepts / adapt_window

                # Multiplicative adaptation bounded to avoid extreme jumps
                if accept_rate > target_accept + 0.05:
                    scale *= 1.1
                elif accept_rate < target_accept - 0.05:
                    scale *= 0.9

                # Hard bounds to prevent degenerate behaviour
                scale = np.clip(scale, 1e-6, 10.0)

                logger.debug(
                    "Chain seed=%d  warmup step %d/%d  accept=%.2f  scale=%.4f",
                    seed, step + 1, self._n_warmup, accept_rate, scale,
                )
                window_accepts = 0

            # Store post-warmup samples
            if step >= self._n_warmup:
                samples[step - self._n_warmup] = params

        # Apply thinning
        thinned = samples[:: self._thin]

        accept_rate = total_accepts / self._n_draws if self._n_draws > 0 else 0.0
        logger.info(
            "Chain seed=%d complete  kept=%d  post-warmup accept=%.3f  "
            "final_scale=%.4f",
            seed, len(thinned), accept_rate, scale,
        )
        return thinned

    # ---- effective sample size (batch-means estimator) -------------------- #

    @staticmethod
    def _compute_ess(samples: np.ndarray) -> float:
        """
        Estimate effective sample size using a batch-means approach.

        Operates on a 1-D array of scalar samples.  Divides the chain into
        batches of size √n, computes batch means, and uses the ratio of
        naïve variance to batch-means variance as the ESS estimate.
        """
        n = len(samples)
        if n < 4:
            return float(n)

        batch_size = max(1, int(np.sqrt(n)))
        n_batches = n // batch_size
        if n_batches < 2:
            return float(n)

        # Trim to exact multiple of batch_size
        trimmed = samples[: n_batches * batch_size]
        batches = trimmed.reshape(n_batches, batch_size)
        batch_means = batches.mean(axis=1)

        var_overall = np.var(samples, ddof=1)
        var_batch_means = np.var(batch_means, ddof=1)

        if var_batch_means < 1e-15:
            return float(n)

        # ESS ≈ n * (var_overall / (batch_size * var_batch_means))
        ess = n * var_overall / (batch_size * var_batch_means)
        return float(np.clip(ess, 1.0, n))

    # ---- softmax helper -------------------------------------------------- #

    @staticmethod
    def _softmax_probs(params: np.ndarray, x: np.ndarray, n_features: int) -> np.ndarray:
        """
        Compute softmax probabilities for a single observation.

        Parameters
        ----------
        params : ndarray, shape (2 * n_features,)
        x : ndarray, shape (n_features,)
        n_features : int

        Returns
        -------
        ndarray, shape (3,)
            [P(home), P(draw), P(away)]
        """
        beta = params[:n_features]
        gamma = params[n_features:]

        eta_home = float(x @ beta)
        eta_away = float(x @ gamma)

        # Log-sum-exp for numerical stability
        max_eta = max(eta_home, eta_away, 0.0)
        denom = np.exp(-max_eta) + np.exp(eta_home - max_eta) + np.exp(eta_away - max_eta)

        p_home = np.exp(eta_home - max_eta) / denom
        p_draw = np.exp(-max_eta) / denom
        p_away = np.exp(eta_away - max_eta) / denom

        return np.array([p_home, p_draw, p_away])

    @staticmethod
    def _softmax_probs_batch(
        samples: np.ndarray, x: np.ndarray, n_features: int
    ) -> np.ndarray:
        """
        Vectorised softmax over all posterior samples for one observation.

        Parameters
        ----------
        samples : ndarray, shape (n_samples, 2 * n_features)
        x : ndarray, shape (n_features,)
        n_features : int

        Returns
        -------
        ndarray, shape (n_samples, 3)
            Columns: [P(home), P(draw), P(away)]
        """
        betas = samples[:, :n_features]   # (n_samples, n_features)
        gammas = samples[:, n_features:]  # (n_samples, n_features)

        eta_home = betas @ x   # (n_samples,)
        eta_away = gammas @ x  # (n_samples,)

        # Log-sum-exp trick
        zeros = np.zeros_like(eta_home)
        max_eta = np.maximum(np.maximum(eta_home, eta_away), zeros)

        exp_home = np.exp(eta_home - max_eta)
        exp_draw = np.exp(zeros - max_eta)
        exp_away = np.exp(eta_away - max_eta)
        denom = exp_home + exp_draw + exp_away

        p_home = exp_home / denom
        p_draw = exp_draw / denom
        p_away = exp_away / denom

        return np.column_stack([p_home, p_draw, p_away])

    # ---- training -------------------------------------------------------- #

    def fit(
        self,
        matches: pd.DataFrame,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        """
        Train MCMC model on recent *matches*.

        Returns a dict of training metrics:
            accuracy, n_samples, acceptance_rate, ess, n_train, n_val
        """
        t0 = time.perf_counter()

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
            "Training MCMCPredictor  n_train=%d  n_val=%d  features=%d  "
            "warmup=%d  draws=%d  chains=%d  thin=%d",
            len(X_train),
            len(X_val),
            X_train.shape[1],
            self._n_warmup,
            self._n_draws,
            self._n_chains,
            self._thin,
        )

        # ---- run chains ----
        chain_samples: list[np.ndarray] = []
        base_seed = 42

        for chain_idx in range(self._n_chains):
            chain_seed = base_seed + chain_idx * 1000
            logger.info("Starting chain %d/%d  seed=%d", chain_idx + 1, self._n_chains, chain_seed)
            samples = self._run_chain(X_train_scaled, y_train, seed=chain_seed)
            chain_samples.append(samples)

        # Concatenate all chains
        self._posterior_samples = np.concatenate(chain_samples, axis=0)
        self._is_fitted = True
        n_total_samples = len(self._posterior_samples)

        elapsed = time.perf_counter() - t0
        logger.info(
            "MCMC sampling complete  total_samples=%d  elapsed=%.1fs",
            n_total_samples, elapsed,
        )

        # ---- diagnostics ----

        # Per-parameter ESS, report the minimum
        ess_values = [
            self._compute_ess(self._posterior_samples[:, j])
            for j in range(self._posterior_samples.shape[1])
        ]
        min_ess = min(ess_values)
        mean_ess = float(np.mean(ess_values))

        # Rough acceptance rate from chain logs (use posterior mean spread as proxy)
        # We already logged per-chain; compute an aggregate from the samples
        # by checking how many consecutive samples differ
        n_unique = sum(
            1
            for i in range(1, n_total_samples)
            if not np.array_equal(
                self._posterior_samples[i], self._posterior_samples[i - 1]
            )
        )
        acceptance_rate = n_unique / max(n_total_samples - 1, 1)

        # ---- validation accuracy using posterior mean ----
        mean_params = self._posterior_samples.mean(axis=0)

        val_preds = np.empty(len(y_val), dtype=int)
        for i in range(len(y_val)):
            probs = self._softmax_probs(mean_params, X_val_scaled[i], self._n_features)
            val_preds[i] = int(np.argmax(probs))

        accuracy = float(np.mean(val_preds == y_val))

        metrics: dict[str, Any] = {
            "accuracy": accuracy,
            "n_samples": n_total_samples,
            "acceptance_rate": round(acceptance_rate, 4),
            "ess": round(min_ess, 1),
            "ess_mean": round(mean_ess, 1),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "elapsed_seconds": round(elapsed, 2),
        }
        logger.info("MCMCPredictor training complete: %s", metrics)
        return metrics

    # ---- point prediction ------------------------------------------------ #

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

        Uses the posterior mean of the MCMC samples.

        Raises RuntimeError if the model has not been fitted.
        """
        if not self._is_fitted or self._posterior_samples is None:
            raise RuntimeError(
                "MCMCPredictor has not been fitted yet.  Call fit() first."
            )

        x_scaled = self._build_single_feature_vector(
            home_team, away_team, rankings, dc_params, neutral
        )

        mean_params = self._posterior_samples.mean(axis=0)
        probs = self._softmax_probs(mean_params, x_scaled, self._n_features)

        p_home = float(probs[0])
        p_draw = float(probs[1])
        p_away = float(probs[2])

        logger.debug(
            "Predict %s vs %s (neutral=%s): H=%.3f D=%.3f A=%.3f",
            home_team, away_team, neutral, p_home, p_draw, p_away,
        )
        return (p_home, p_draw, p_away)

    # ---- prediction with uncertainty ------------------------------------- #

    def predict_with_uncertainty(
        self,
        home_team: str,
        away_team: str,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None = None,
        neutral: bool = True,
    ) -> dict[str, tuple[float, float, float]]:
        """
        Predict match probabilities with full posterior uncertainty.

        For each posterior sample, the softmax probabilities are computed,
        giving a distribution over (P_home, P_draw, P_away).

        Returns
        -------
        dict with keys:
            "mean"     — posterior mean (p_home, p_draw, p_away)
            "lower_5"  — 5th percentile
            "upper_95" — 95th percentile
            "std"      — posterior standard deviation
        """
        if not self._is_fitted or self._posterior_samples is None:
            raise RuntimeError(
                "MCMCPredictor has not been fitted yet.  Call fit() first."
            )

        x_scaled = self._build_single_feature_vector(
            home_team, away_team, rankings, dc_params, neutral
        )

        # Vectorised computation over all posterior samples
        all_probs = self._softmax_probs_batch(
            self._posterior_samples, x_scaled, self._n_features
        )  # (n_samples, 3)

        mean_probs = all_probs.mean(axis=0)
        std_probs = all_probs.std(axis=0)
        lower_5 = np.percentile(all_probs, 5, axis=0)
        upper_95 = np.percentile(all_probs, 95, axis=0)

        result = {
            "mean": (float(mean_probs[0]), float(mean_probs[1]), float(mean_probs[2])),
            "lower_5": (float(lower_5[0]), float(lower_5[1]), float(lower_5[2])),
            "upper_95": (float(upper_95[0]), float(upper_95[1]), float(upper_95[2])),
            "std": (float(std_probs[0]), float(std_probs[1]), float(std_probs[2])),
        }

        logger.debug(
            "Uncertainty for %s vs %s: mean=(%.3f, %.3f, %.3f)  "
            "std=(%.3f, %.3f, %.3f)",
            home_team,
            away_team,
            *result["mean"],
            *result["std"],
        )
        return result

    # ---- internal helpers ------------------------------------------------- #

    def _build_single_feature_vector(
        self,
        home_team: str,
        away_team: str,
        rankings: dict[str, int],
        dc_params: dict[str, dict[str, float]] | None,
        neutral: bool,
    ) -> np.ndarray:
        """
        Build and scale a single-row feature vector for prediction.

        Returns
        -------
        ndarray, shape (n_features,)
            Scaled feature vector ready for multiplication with params.
        """
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
        return row_scaled[0]  # flatten to 1-D
