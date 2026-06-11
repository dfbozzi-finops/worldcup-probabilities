"""
World Cup Backtesting Harness — Statistical Arbitrage System

Replays the Dixon-Coles + Monte Carlo pipeline against historical World Cups
to validate model calibration and measure hypothetical P&L.

Currently supports the 2022 World Cup (32-team / 8-group format).
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.data_loader import load_match_data, get_fifa_rankings, normalize_team_name
from src.dixon_coles import DixonColesModel

logger = logging.getLogger(__name__)

# ======================================================================
# 2022 R16 bracket definition (standard FIFA 32-team format)
#
# Upper half:  1A-2B, 1C-2D, 1E-2F, 1G-2H
# Lower half:  1B-2A, 1D-2C, 1F-2E, 1H-2G
#
# QF pairings:
#   W(1A-2B) vs W(1C-2D)     W(1E-2F) vs W(1G-2H)
#   W(1B-2A) vs W(1D-2C)     W(1F-2E) vs W(1H-2G)
#
# SF pairings:
#   W(QF1) vs W(QF2)         W(QF3) vs W(QF4)
# ======================================================================

R16_BRACKET_32: list[tuple[str, str]] = [
    # Upper half
    ("1A", "2B"),  # R16-1
    ("1C", "2D"),  # R16-2
    ("1E", "2F"),  # R16-3
    ("1G", "2H"),  # R16-4
    # Lower half
    ("1B", "2A"),  # R16-5
    ("1D", "2C"),  # R16-6
    ("1F", "2E"),  # R16-7
    ("1H", "2G"),  # R16-8
]

QF_BRACKET_32: list[tuple[int, int]] = [
    (0, 1),  # QF1: W(R16-1) vs W(R16-2)
    (2, 3),  # QF2: W(R16-3) vs W(R16-4)
    (4, 5),  # QF3: W(R16-5) vs W(R16-6)
    (6, 7),  # QF4: W(R16-7) vs W(R16-8)
]

SF_BRACKET_32: list[tuple[int, int]] = [
    (0, 1),  # SF1: W(QF1) vs W(QF2)
    (2, 3),  # SF2: W(QF3) vs W(QF4)
]


# ======================================================================
# Tournament start dates (for filtering pre-tournament data)
# ======================================================================

TOURNAMENT_START_DATES: dict[int, str] = {
    2022: "2022-11-20",
}

# Approximate pre-tournament bookmaker odds for the eventual winner
# Used for hypothetical P&L calculation
BOOKMAKER_MEDIAN_ODDS: dict[int, dict[str, float]] = {
    2022: {
        "Argentina": 5.0,
        "Brazil": 4.0,
        "France": 6.5,
        "England": 8.0,
        "Spain": 8.0,
        "Germany": 10.0,
        "Netherlands": 12.0,
        "Portugal": 12.0,
        "Belgium": 16.0,
        "Denmark": 20.0,
        "Croatia": 25.0,
        "Uruguay": 30.0,
        "Switzerland": 50.0,
        "Senegal": 80.0,
        "United States": 100.0,
        "Mexico": 100.0,
        "Poland": 100.0,
        "Morocco": 150.0,
        "Japan": 150.0,
        "South Korea": 200.0,
        "Australia": 200.0,
        "Iran": 250.0,
        "Ecuador": 150.0,
        "Qatar": 200.0,
        "Cameroon": 200.0,
        "Ghana": 200.0,
        "Tunisia": 250.0,
        "Saudi Arabia": 300.0,
        "Wales": 200.0,
        "Canada": 150.0,
        "Costa Rica": 500.0,
        "Serbia": 80.0,
    },
}


class WorldCupBacktester:
    """
    Backtesting harness that replays the Dixon-Coles + Monte Carlo pipeline
    against historical World Cup tournaments.

    Parameters
    ----------
    settings : dict
        Application settings (same format as config/settings.json).
    n_simulations : int
        Number of Monte Carlo iterations for tournament simulation.
    """

    def __init__(self, settings: dict, n_simulations: int = 10_000) -> None:
        self.settings = settings
        self.n_simulations = n_simulations

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_tournament(self, year: int) -> dict:
        """
        Load ``data/historical/wc_{year}.json`` and return the full
        tournament structure including groups, bracket, and actual results.
        """
        path = Path(f"data/historical/wc_{year}.json")
        if not path.exists():
            raise FileNotFoundError(
                f"Tournament data file not found: {path}. "
                f"Ensure data/historical/wc_{year}.json exists."
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.info(
            "Loaded %d World Cup data — host: %s, winner: %s",
            data["year"],
            data["host"],
            data["winner"],
        )
        return data

    # ------------------------------------------------------------------
    # Model fitting (pre-tournament data only)
    # ------------------------------------------------------------------

    def _fit_model_at_cutoff(self, tournament: dict) -> DixonColesModel:
        """
        Fit the Dixon-Coles model using only matches that occurred
        BEFORE the tournament start date, ensuring no data leakage.

        Parameters
        ----------
        tournament : dict
            Tournament data loaded from ``_load_tournament``.

        Returns
        -------
        DixonColesModel
            Fitted model using only pre-tournament data.
        """
        year = tournament["year"]
        start_date = tournament.get("start_date", TOURNAMENT_START_DATES[year])

        dc_settings = self.settings.get("dixon_coles", {})
        history_years = dc_settings.get("history_years", 20)

        logger.info(
            "Loading match data — filtering to before %s (history_years=%d)",
            start_date,
            history_years,
        )

        df = load_match_data(history_years=history_years)

        # Filter to matches BEFORE the tournament start date
        cutoff = df["date"] < start_date
        df_pre = df.loc[cutoff].copy()

        logger.info(
            "Pre-tournament dataset: %d matches (from %s to %s)",
            len(df_pre),
            df_pre["date"].min().strftime("%Y-%m-%d") if len(df_pre) > 0 else "N/A",
            df_pre["date"].max().strftime("%Y-%m-%d") if len(df_pre) > 0 else "N/A",
        )

        # Fit the Dixon-Coles model
        # For the 2022 WC the host was Qatar
        host_nations = [tournament.get("host", "Qatar")]
        model = DixonColesModel()
        model.fit(df_pre, host_nations=host_nations)

        return model

    # ------------------------------------------------------------------
    # Group stage simulation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_group_match(
        predict_fn: Callable[[str, str], tuple[float, float, float]],
        team_a: str,
        team_b: str,
    ) -> tuple[int, int]:
        """
        Simulate a single group-stage match and return (goals_a, goals_b).

        Uses simplified scorelines: win → 2-0, draw → 1-1, loss → 0-2.
        This matches the 2026 simulator logic.
        """
        p_a, p_draw, p_b = predict_fn(team_a, team_b)
        total = p_a + p_draw + p_b
        if total <= 0:
            p_a, p_draw, p_b = 1 / 3, 1 / 3, 1 / 3
            total = 1.0

        p_a /= total
        p_draw /= total

        r = random.random()
        if r < p_a:
            return (2, 0)
        elif r < p_a + p_draw:
            return (1, 1)
        else:
            return (0, 2)

    @staticmethod
    def _simulate_knockout_match(
        predict_fn: Callable[[str, str], tuple[float, float, float]],
        team_a: str,
        team_b: str,
    ) -> str:
        """
        Simulate a knockout match and return the winning team.

        If the model predicts a draw, a 50/50 coin flip decides
        (representing a penalty shoot-out).
        """
        p_a, p_draw, p_b = predict_fn(team_a, team_b)
        total = p_a + p_draw + p_b
        if total <= 0:
            p_a, p_draw, p_b = 1 / 3, 1 / 3, 1 / 3
            total = 1.0

        p_a /= total
        p_draw /= total

        r = random.random()
        if r < p_a:
            return team_a
        elif r < p_a + p_draw:
            return team_a if random.random() < 0.5 else team_b
        else:
            return team_b

    @staticmethod
    def _simulate_group(
        predict_fn: Callable[[str, str], tuple[float, float, float]],
        teams: list[str],
    ) -> list[str]:
        """
        Simulate a round-robin group of 4 teams.

        Returns the teams sorted by final standing:
        [1st, 2nd, 3rd, 4th] using tiebreakers: points > GD > GF.
        """
        records: dict[str, dict[str, int]] = {
            t: {"pts": 0, "gd": 0, "gf": 0} for t in teams
        }

        for t_a, t_b in combinations(teams, 2):
            ga, gb = WorldCupBacktester._simulate_group_match(
                predict_fn, t_a, t_b
            )
            records[t_a]["gf"] += ga
            records[t_b]["gf"] += gb
            records[t_a]["gd"] += ga - gb
            records[t_b]["gd"] += gb - ga

            if ga > gb:
                records[t_a]["pts"] += 3
            elif ga == gb:
                records[t_a]["pts"] += 1
                records[t_b]["pts"] += 1
            else:
                records[t_b]["pts"] += 3

        ranking = sorted(
            teams,
            key=lambda t: (
                records[t]["pts"],
                records[t]["gd"],
                records[t]["gf"],
            ),
            reverse=True,
        )

        return ranking

    # ------------------------------------------------------------------
    # 32-team tournament simulation
    # ------------------------------------------------------------------

    def _simulate_tournament_32(
        self,
        predict_fn: Callable[[str, str], tuple[float, float, float]],
        tournament: dict,
    ) -> dict[str, float]:
        """
        Simulate a standard 32-team / 8-group World Cup format using
        Monte Carlo and return team → win_probability.

        Bracket structure (2022 format):
          R16 upper: 1A-2B, 1C-2D, 1E-2F, 1G-2H
          R16 lower: 1B-2A, 1D-2C, 1F-2E, 1H-2G
          QF: W(1A-2B) vs W(1C-2D), W(1E-2F) vs W(1G-2H),
              W(1B-2A) vs W(1D-2C), W(1F-2E) vs W(1H-2G)
          SF: W(QF1) vs W(QF2), W(QF3) vs W(QF4)
          Final: W(SF1) vs W(SF2)

        Parameters
        ----------
        predict_fn : callable
            ``(team_a, team_b) → (p_a_win, p_draw, p_b_win)``
        tournament : dict
            Tournament data with ``groups`` key.

        Returns
        -------
        dict[str, float]
            ``{team_name: win_probability}`` sorted descending.
        """
        groups = tournament["groups"]
        wins: dict[str, int] = defaultdict(int)

        for i in range(1, self.n_simulations + 1):
            # --- Group stage ---
            group_standings: dict[str, list[str]] = {}
            for letter, teams in groups.items():
                ranking = self._simulate_group(predict_fn, list(teams))
                group_standings[letter] = ranking

            # --- R16 ---
            r16_winners: list[str] = []
            for slot_home, slot_away in R16_BRACKET_32:
                # Parse slot codes like "1A" → 1st in group A, "2B" → 2nd in B
                pos_h = int(slot_home[0]) - 1  # 0-indexed
                grp_h = slot_home[1]
                pos_a = int(slot_away[0]) - 1
                grp_a = slot_away[1]

                team_h = group_standings[grp_h][pos_h]
                team_a = group_standings[grp_a][pos_a]
                winner = self._simulate_knockout_match(predict_fn, team_h, team_a)
                r16_winners.append(winner)

            # --- QF ---
            qf_winners: list[str] = []
            for idx_a, idx_b in QF_BRACKET_32:
                winner = self._simulate_knockout_match(
                    predict_fn, r16_winners[idx_a], r16_winners[idx_b]
                )
                qf_winners.append(winner)

            # --- SF ---
            sf_winners: list[str] = []
            for idx_a, idx_b in SF_BRACKET_32:
                winner = self._simulate_knockout_match(
                    predict_fn, qf_winners[idx_a], qf_winners[idx_b]
                )
                sf_winners.append(winner)

            # --- Final ---
            champion = self._simulate_knockout_match(
                predict_fn, sf_winners[0], sf_winners[1]
            )
            wins[champion] += 1

            if i % 2000 == 0:
                logger.info(
                    "[Backtest Simulator] %d/%d iterations complete",
                    i,
                    self.n_simulations,
                )

        # Convert to probabilities
        probs: dict[str, float] = {
            team: count / self.n_simulations for team, count in wins.items()
        }

        # Ensure all 32 teams appear (even if they never won)
        all_teams = [t for g in groups.values() for t in g]
        for team in all_teams:
            if team not in probs:
                probs[team] = 0.0

        # Sort descending
        probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))

        return probs

    # ------------------------------------------------------------------
    # Backtest execution
    # ------------------------------------------------------------------

    def run_backtest(self, year: int = 2022) -> dict:
        """
        Run a complete backtest for the specified World Cup year.

        Steps:
        1. Load tournament data (groups, bracket, actual results).
        2. Fit Dixon-Coles on pre-tournament-only historical data.
        3. Run Monte Carlo simulation using the fitted model.
        4. Compare model probabilities against actual outcomes.
        5. Calculate calibration metrics and hypothetical P&L.

        Parameters
        ----------
        year : int
            World Cup year to backtest (currently only 2022 supported).

        Returns
        -------
        dict
            Comprehensive results including Brier score, log-loss,
            top-K accuracy, winner rank, predicted ranking, and
            simulated P&L.
        """
        logger.info("=" * 60)
        logger.info("Starting backtest for %d World Cup", year)
        logger.info("=" * 60)

        # Step 1: Load tournament data
        tournament = self._load_tournament(year)
        actual_winner = tournament["winner"]

        # Step 2: Fit Dixon-Coles on pre-tournament data
        model = self._fit_model_at_cutoff(tournament)

        # Step 3: Build predict_fn from the fitted model
        def predict_fn(team_a: str, team_b: str) -> tuple[float, float, float]:
            a = normalize_team_name(team_a)
            b = normalize_team_name(team_b)
            return model.predict_match(a, b, neutral=True)

        # Step 4: Run Monte Carlo simulation
        logger.info(
            "Running %d Monte Carlo simulations (32-team format)…",
            self.n_simulations,
        )
        probs = self._simulate_tournament_32(predict_fn, tournament)

        # Step 5: Calculate metrics
        sorted_teams = sorted(probs.items(), key=lambda kv: -kv[1])
        predicted_ranking = [(team, prob) for team, prob in sorted_teams]

        # Winner probability and rank
        winner_prob = probs.get(actual_winner, 0.0)
        winner_rank = next(
            (i + 1 for i, (team, _) in enumerate(predicted_ranking) if team == actual_winner),
            len(predicted_ranking),
        )

        # Brier score: mean((p_i - actual_i)^2) across all 32 teams
        all_teams = [t for g in tournament["groups"].values() for t in g]
        brier_scores = []
        for team in all_teams:
            predicted = probs.get(team, 0.0)
            actual = 1.0 if team == actual_winner else 0.0
            brier_scores.append((predicted - actual) ** 2)
        brier_score = float(np.mean(brier_scores))

        # Log-loss: -log(P_winner)
        log_loss = -math.log(max(winner_prob, 1e-10))

        # Top-K accuracy
        top_teams = [team for team, _ in predicted_ranking]
        top_k_accuracy = {
            3: actual_winner in top_teams[:3],
            5: actual_winner in top_teams[:5],
            10: actual_winner in top_teams[:10],
        }

        # Simulated P&L using Kelly criterion
        simulated_pnl = self._calculate_pnl(
            probs, actual_winner, year
        )

        results = {
            "year": year,
            "actual_winner": actual_winner,
            "runner_up": tournament.get("runner_up", ""),
            "third": tournament.get("third", ""),
            "fourth": tournament.get("fourth", ""),
            "brier_score": brier_score,
            "log_loss": log_loss,
            "top_k_accuracy": top_k_accuracy,
            "winner_prob": winner_prob,
            "winner_rank": winner_rank,
            "predicted_ranking": predicted_ranking,
            "simulated_pnl": simulated_pnl,
            "n_simulations": self.n_simulations,
        }

        logger.info("Backtest complete for %d World Cup", year)
        return results

    # ------------------------------------------------------------------
    # P&L calculation
    # ------------------------------------------------------------------

    def _calculate_pnl(
        self,
        probs: dict[str, float],
        actual_winner: str,
        year: int,
    ) -> dict:
        """
        Calculate hypothetical P&L using Kelly criterion against
        pre-tournament bookmaker odds.

        Assumes a $10,000 bankroll and quarter-Kelly sizing (matching
        the system's production settings).

        Parameters
        ----------
        probs : dict[str, float]
            Model-predicted win probabilities for each team.
        actual_winner : str
            The team that actually won.
        year : int
            Tournament year (used to look up bookmaker odds).

        Returns
        -------
        dict
            P&L breakdown including bets placed, returns, and ROI.
        """
        bankroll = 10_000.0
        kelly_multiplier = self.settings.get("kelly_fraction", 0.25)
        max_position_pct = self.settings.get("max_position_pct", 0.05)
        ev_threshold = self.settings.get("ev_threshold", 0.02)

        odds_table = BOOKMAKER_MEDIAN_ODDS.get(year, {})

        bets: list[dict] = []
        total_wagered = 0.0
        total_return = 0.0

        for team, model_prob in probs.items():
            market_odds = odds_table.get(team)
            if market_odds is None:
                continue

            market_prob = 1.0 / market_odds
            ev = model_prob * (market_odds - 1.0) - (1.0 - model_prob)

            if ev <= ev_threshold:
                continue
            if model_prob < 0.02:
                continue

            # Kelly sizing
            denom = market_odds - 1.0
            if denom <= 0:
                continue

            kelly_full = (model_prob * market_odds - 1.0) / denom
            kelly_frac = kelly_full * kelly_multiplier
            position_frac = min(max(kelly_frac, 0.0), max_position_pct)
            wager = position_frac * bankroll

            won = team == actual_winner
            payout = wager * market_odds if won else 0.0
            profit = payout - wager

            bets.append({
                "team": team,
                "model_prob": round(model_prob, 4),
                "market_odds": market_odds,
                "market_prob": round(market_prob, 4),
                "ev": round(ev, 4),
                "kelly_frac": round(kelly_frac, 4),
                "wager": round(wager, 2),
                "won": won,
                "payout": round(payout, 2),
                "profit": round(profit, 2),
            })

            total_wagered += wager
            total_return += payout

        net_pnl = total_return - total_wagered
        roi = (net_pnl / total_wagered * 100) if total_wagered > 0 else 0.0

        return {
            "bankroll": bankroll,
            "n_bets": len(bets),
            "total_wagered": round(total_wagered, 2),
            "total_return": round(total_return, 2),
            "net_pnl": round(net_pnl, 2),
            "roi_pct": round(roi, 2),
            "bets": bets,
        }

    # ------------------------------------------------------------------
    # Rich report
    # ------------------------------------------------------------------

    def print_report(self, results: dict, console: Console | None = None) -> None:
        """
        Print a rich-formatted backtest report.

        Parameters
        ----------
        results : dict
            Output from ``run_backtest()``.
        console : Console, optional
            Rich Console instance. Created if not provided.
        """
        if console is None:
            console = Console()

        year = results["year"]
        actual_winner = results["actual_winner"]

        # ── Header ──
        console.print()
        console.print(
            Panel(
                f"[bold cyan]World Cup {year} Backtest Report[/bold cyan]\n"
                f"[dim]Monte Carlo simulations: {results['n_simulations']:,}[/dim]",
                border_style="bright_blue",
                expand=False,
            )
        )

        # ── Calibration Metrics ──
        metrics_table = Table(
            title="📊 Calibration Metrics",
            title_style="bold yellow",
            show_header=True,
            header_style="bold magenta",
        )
        metrics_table.add_column("Metric", style="cyan", min_width=25)
        metrics_table.add_column("Value", style="green", justify="right")

        metrics_table.add_row("Actual Winner", f"🏆 {actual_winner}")
        metrics_table.add_row("Winner Probability", f"{results['winner_prob']:.4f} ({results['winner_prob']*100:.2f}%)")
        metrics_table.add_row("Winner Rank", f"#{results['winner_rank']} of 32")
        metrics_table.add_row("Brier Score", f"{results['brier_score']:.6f}")
        metrics_table.add_row("Log-Loss", f"{results['log_loss']:.4f}")

        top_k = results["top_k_accuracy"]
        for k in [3, 5, 10]:
            status = "✅" if top_k[k] else "❌"
            metrics_table.add_row(f"Top-{k} Accuracy", f"{status}")

        console.print(metrics_table)
        console.print()

        # ── Top-10 Predicted Rankings ──
        ranking_table = Table(
            title="🏆 Top-10 Predicted Rankings vs Actual",
            title_style="bold yellow",
            show_header=True,
            header_style="bold magenta",
        )
        ranking_table.add_column("#", style="dim", width=4, justify="right")
        ranking_table.add_column("Team", style="cyan", min_width=20)
        ranking_table.add_column("Win Prob", style="green", justify="right")
        ranking_table.add_column("Actual", style="yellow", justify="center")

        # Actual final standings for highlighting
        actual_standings = {
            results["actual_winner"]: "🥇 Winner",
            results.get("runner_up", ""): "🥈 Runner-up",
            results.get("third", ""): "🥉 Third",
            results.get("fourth", ""): "4th",
        }

        for i, (team, prob) in enumerate(results["predicted_ranking"][:10]):
            rank_str = str(i + 1)
            prob_str = f"{prob:.4f} ({prob*100:.2f}%)"
            actual_str = actual_standings.get(team, "—")
            style = "bold green" if team == actual_winner else None
            ranking_table.add_row(rank_str, team, prob_str, actual_str, style=style)

        console.print(ranking_table)
        console.print()

        # ── Simulated P&L ──
        pnl = results["simulated_pnl"]
        pnl_color = "green" if pnl["net_pnl"] >= 0 else "red"

        pnl_table = Table(
            title="💰 Simulated P&L (Quarter-Kelly)",
            title_style="bold yellow",
            show_header=True,
            header_style="bold magenta",
        )
        pnl_table.add_column("Metric", style="cyan", min_width=20)
        pnl_table.add_column("Value", style=pnl_color, justify="right")

        pnl_table.add_row("Bankroll", f"${pnl['bankroll']:,.2f}")
        pnl_table.add_row("Bets Placed", str(pnl["n_bets"]))
        pnl_table.add_row("Total Wagered", f"${pnl['total_wagered']:,.2f}")
        pnl_table.add_row("Total Return", f"${pnl['total_return']:,.2f}")
        pnl_table.add_row(
            "Net P&L",
            f"${pnl['net_pnl']:+,.2f}",
        )
        pnl_table.add_row("ROI", f"{pnl['roi_pct']:+.2f}%")

        console.print(pnl_table)

        # ── Individual Bets ──
        if pnl["bets"]:
            console.print()
            bets_table = Table(
                title="📋 Individual Bets",
                title_style="bold yellow",
                show_header=True,
                header_style="bold magenta",
            )
            bets_table.add_column("Team", style="cyan", min_width=15)
            bets_table.add_column("Model P", justify="right")
            bets_table.add_column("Mkt Odds", justify="right")
            bets_table.add_column("EV", justify="right")
            bets_table.add_column("Wager", justify="right")
            bets_table.add_column("Won?", justify="center")
            bets_table.add_column("Profit", justify="right")

            for bet in sorted(pnl["bets"], key=lambda b: -b["wager"]):
                won_str = "✅" if bet["won"] else "❌"
                profit_str = f"${bet['profit']:+,.2f}"
                profit_style = "green" if bet["profit"] >= 0 else "red"
                bets_table.add_row(
                    bet["team"],
                    f"{bet['model_prob']:.4f}",
                    f"{bet['market_odds']:.1f}x",
                    f"{bet['ev']:.4f}",
                    f"${bet['wager']:,.2f}",
                    won_str,
                    Text(profit_str, style=profit_style),
                )

            console.print(bets_table)

        console.print()
        console.print(
            Panel(
                f"[dim]Backtest complete. Brier score of "
                f"{results['brier_score']:.6f} — "
                f"{'lower is better' if results['brier_score'] < 0.05 else 'room for improvement'}. "
                f"Model ranked {actual_winner} #{results['winner_rank']}.[/dim]",
                border_style="dim",
                expand=False,
            )
        )
