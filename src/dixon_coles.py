"""
Dixon-Coles Model — World Cup 2026 Statistical Arbitrage System

Implements the Dixon & Coles (1997) bivariate Poisson model with:
  • team-specific attack / defence parameters,
  • home-advantage term (with host-nation logic for World Cups),
  • ρ (rho) correction for low-scoring outcomes,
  • exponential time-decay weighting.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from src.data_loader import normalize_team_name

if TYPE_CHECKING:
    import pandas as pd


class DixonColesModel:
    """
    Fit and predict with the Dixon-Coles bivariate Poisson model.

    After calling :meth:`fit`, use :meth:`predict_match` or
    :meth:`predict_score_probs` to generate match-outcome probabilities.
    """

    def __init__(self) -> None:
        self.attack: dict[str, float] = {}
        self.defence: dict[str, float] = {}
        self.home_adv: float = 0.0
        self.rho: float = 0.0
        self.teams: list[str] = []
        self._avg_attack: float = 0.0
        self._avg_defence: float = 0.0

    # ------------------------------------------------------------------
    # Tau correction (Dixon-Coles low-score adjustment)
    # ------------------------------------------------------------------

    @staticmethod
    def _tau(
        x: int,
        y: int,
        lambda_val: float,
        mu_val: float,
        rho: float,
    ) -> float:
        """Return the τ correction factor for score (x, y)."""
        if x == 0 and y == 0:
            return 1.0 - lambda_val * mu_val * rho
        if x == 0 and y == 1:
            return 1.0 + lambda_val * rho
        if x == 1 and y == 0:
            return 1.0 + mu_val * rho
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    # ------------------------------------------------------------------
    # Negative log-likelihood
    # ------------------------------------------------------------------

    @staticmethod
    def _neg_log_likelihood(
        params: np.ndarray,
        matches: np.ndarray,
        n_teams: int,
        weights: np.ndarray,
        host_indices: set[int],
    ) -> float:
        """
        Compute the negative log-likelihood for the Dixon-Coles model.

        Parameters layout
        -----------------
        params[0 .. N-1]         : attack_i   (one per team)
        params[N .. 2N-2]        : defence_i  (N-1 values; last = −sum)
        params[2N-1]             : home_adv
        params[2N]               : rho
        """
        attacks = params[:n_teams]
        defences_partial = params[n_teams : 2 * n_teams - 1]
        # Identifiability constraint: sum of defence params = 0
        last_defence = -defences_partial.sum()
        defences = np.append(defences_partial, last_defence)
        home_adv = params[2 * n_teams - 1]
        rho = params[2 * n_teams]

        home_idx = matches[:, 0].astype(int)
        away_idx = matches[:, 1].astype(int)
        home_goals = matches[:, 2].astype(int)
        away_goals = matches[:, 3].astype(int)
        neutral = matches[:, 4].astype(int)

        host_mask = np.isin(home_idx, list(host_indices))
        home_factor = np.where(neutral == 0, 1.0, np.where(host_mask, 0.5, 0.0))

        lambda_val = np.exp(attacks[home_idx] + defences[away_idx] + home_adv * home_factor)
        mu_val = np.exp(attacks[away_idx] + defences[home_idx])

        lambda_val = np.clip(lambda_val, a_min=None, a_max=15.0)
        mu_val = np.clip(mu_val, a_min=None, a_max=15.0)

        tau = np.ones_like(lambda_val)
        
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        
        tau[mask_00] = 1.0 - lambda_val[mask_00] * mu_val[mask_00] * rho
        tau[mask_01] = 1.0 + lambda_val[mask_01] * rho
        tau[mask_10] = 1.0 + mu_val[mask_10] * rho
        tau[mask_11] = 1.0 - rho

        prob = tau * poisson.pmf(home_goals, lambda_val) * poisson.pmf(away_goals, mu_val)
        prob = np.clip(prob, a_min=1e-10, a_max=None)
        
        nll = -np.sum(weights * np.log(prob))

        return float(nll)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        matches: pd.DataFrame,
        host_nations: list[str] | None = None,
    ) -> None:
        """
        Fit the Dixon-Coles model to historical match data.

        Parameters
        ----------
        matches : pd.DataFrame
            Must contain: date, home_team, away_team, home_score,
            away_score, neutral.
        host_nations : list[str], optional
            Teams that are hosting the tournament (receive a partial
            home-advantage on neutral ground).
        """
        if host_nations is None:
            host_nations = []

        # Normalise team names
        df = matches.copy()
        df["home_team"] = df["home_team"].map(normalize_team_name)
        df["away_team"] = df["away_team"].map(normalize_team_name)

        # Build team index
        all_teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        team_to_idx: dict[str, int] = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        host_indices: set[int] = {
            team_to_idx[normalize_team_name(h)]
            for h in host_nations
            if normalize_team_name(h) in team_to_idx
        }

        # Time-decay weights  (half-life ≈ 2 years)
        xi = math.log(2) / (2 * 365.25)
        max_date = df["date"].max()
        days_since = (max_date - df["date"]).dt.days.values.astype(float)
        weights = np.exp(-xi * days_since)

        # Encode matches as a numeric array for speed
        #   columns: home_idx, away_idx, home_score, away_score, neutral
        match_arr = np.column_stack(
            [
                df["home_team"].map(team_to_idx).values,
                df["away_team"].map(team_to_idx).values,
                df["home_score"].values,
                df["away_score"].values,
                df["neutral"].astype(int).values,
            ]
        )

        # Initial parameter vector
        attacks_init = np.zeros(n_teams)
        defences_init = np.zeros(n_teams - 1)
        home_adv_init = np.array([0.25])
        rho_init = np.array([-0.1])
        x0 = np.concatenate(
            [attacks_init, defences_init, home_adv_init, rho_init]
        )

        # Bounds
        attack_bounds = [(-3.0, 3.0)] * n_teams
        defence_bounds = [(-3.0, 3.0)] * (n_teams - 1)
        home_adv_bounds = [(0.0, 1.0)]
        rho_bounds = [(-0.5, 0.5)]
        bounds = attack_bounds + defence_bounds + home_adv_bounds + rho_bounds

        print(
            f"[Dixon-Coles] Fitting on {len(df):,} matches with "
            f"{n_teams} teams …"
        )

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(match_arr, n_teams, weights, host_indices),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "disp": False},
        )

        print(
            f"[Dixon-Coles] Optimisation {'converged' if result.success else 'WARNING: did NOT converge'}  "
            f"(fun={result.fun:.2f}, nit={result.nit})"
        )

        # Unpack parameters
        opt = result.x
        attack_vals = opt[:n_teams]
        defence_partial = opt[n_teams : 2 * n_teams - 1]
        last_def = -defence_partial.sum()
        defence_vals = np.append(defence_partial, last_def)

        self.attack = {t: float(attack_vals[i]) for i, t in enumerate(all_teams)}
        self.defence = {t: float(defence_vals[i]) for i, t in enumerate(all_teams)}
        self.home_adv = float(opt[2 * n_teams - 1])
        self.rho = float(opt[2 * n_teams])
        self.teams = all_teams

        # Store averages for unknown-team fallback
        self._avg_attack = float(np.mean(attack_vals))
        self._avg_defence = float(np.mean(defence_vals))

    # ------------------------------------------------------------------
    # Helpers for prediction
    # ------------------------------------------------------------------

    def _get_params(self, team: str) -> tuple[float, float]:
        """Return (attack, defence) for *team*, falling back to averages."""
        team = normalize_team_name(team)
        att = self.attack.get(team, self._avg_attack)
        dfc = self.defence.get(team, self._avg_defence)
        return att, dfc

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_score_probs(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        max_goals: int = 10,
    ) -> np.ndarray:
        """
        Return a (max_goals × max_goals) matrix of joint score probabilities.

        ``mat[i, j]`` = P(home scores *i*, away scores *j*).
        """
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)

        att_h, def_h = self._get_params(home_team)
        att_a, def_a = self._get_params(away_team)

        ha = 0.0 if neutral else self.home_adv
        
        # Widen variance parameter to flatten expected goals (prevents overfitting)
        variance_factor = 2.0
        lambda_val = min(math.exp((att_h + def_a + ha) / variance_factor), 15.0)
        mu_val = min(math.exp((att_a + def_h) / variance_factor), 15.0)

        # Vectorized probability grid computation
        i_arr = np.arange(max_goals)
        lam_arr = poisson.pmf(i_arr, lambda_val)
        mu_arr = poisson.pmf(i_arr, mu_val)
        
        mat = np.outer(lam_arr, mu_arr)
        
        # Apply tau correction for low scores
        mat[0, 0] *= (1.0 - lambda_val * mu_val * self.rho)
        mat[0, 1] *= (1.0 + lambda_val * self.rho)
        mat[1, 0] *= (1.0 + mu_val * self.rho)
        mat[1, 1] *= (1.0 - self.rho)

        # Normalise so the matrix sums to 1 (truncation correction)
        total = mat.sum()
        if total > 0:
            mat /= total

        return mat

    @functools.lru_cache(maxsize=None)
    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
        max_goals: int = 10,
    ) -> tuple[float, float, float]:
        """
        Return ``(p_home_win, p_draw, p_away_win)`` for a single match.

        Parameters
        ----------
        home_team, away_team : str
            Team names (will be normalised internally).
        neutral : bool
            If *True* the home-advantage term is NOT applied (default for
            World Cup matches on neutral ground).
        max_goals : int
            Upper bound on goals for the probability grid.
        """
        mat = self.predict_score_probs(
            home_team, away_team, neutral=neutral, max_goals=max_goals
        )

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if i > j:
                    p_home += mat[i, j]
                elif i == j:
                    p_draw += mat[i, j]
                else:
                    p_away += mat[i, j]

        return p_home, p_draw, p_away
