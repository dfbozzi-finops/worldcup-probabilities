"""World Cup 2026 — Hybrid Statistical Arbitrage Pipeline.

Master orchestrator that runs the end-to-end pipeline:

1. Load historical match data & FIFA rankings.
2. Fit the Dixon-Coles model.
3. Run Monte Carlo simulation (statistical).
4. Train ML models (CatBoost + Bayesian) & run ML simulation.
5. Fetch live Polymarket odds.
6. Compute consensus probabilities & detect arbitrage opportunities.

Usage::

    python main.py           # Continuous polling mode (default 60-min interval)
    python main.py --once    # Single iteration, no loop
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_settings(path: str = "config/settings.json") -> dict:
    """Load the JSON settings file.

    Raises ``SystemExit`` with a clear message when the file is missing or
    malformed so the user gets actionable feedback instead of a traceback.
    """
    settings_path = Path(path)
    if not settings_path.exists():
        logger.critical("Settings file not found at %s", settings_path.resolve())
        sys.exit(1)
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.critical("Invalid JSON in %s: %s", settings_path, exc)
        sys.exit(1)


def _timestamp() -> str:
    """Return the current UTC time formatted for display."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(settings: dict, console: Console) -> int:
    """Execute a single iteration of the full pipeline.

    Returns the number of arbitrage opportunities detected so the caller
    can decide how to proceed.
    """
    bankroll = float(os.getenv("BANKROLL", "10000"))

    # ── Step 1: Historical data ─────────────────────────────────────
    console.print(f"\n[bold cyan]Step 1/6:[/] Loading historical match data …  [dim]{_timestamp()}[/]")
    from src.data_loader import load_match_data, get_fifa_rankings

    history_years = settings.get("dixon_coles", {}).get("history_years", 20)
    matches = load_match_data(history_years)
    rankings = get_fifa_rankings()
    console.print(f"  ✓ Loaded [bold]{len(matches):,}[/] matches, "
                  f"[bold]{len(rankings)}[/] FIFA rankings")

    # ── Step 2: Dixon-Coles ─────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 2/6:[/] Fitting Dixon-Coles model …  [dim]{_timestamp()}[/]")
    from src.dixon_coles import DixonColesModel

    dc = DixonColesModel()
    host_nations = settings.get("host_nations", [])
    dc.fit(matches, host_nations=host_nations)
    console.print(f"  ✓ Fitted [bold]{len(dc.attack)}[/] teams, "
                  f"ρ = {dc.rho:.4f}")

    # ── Step 3: Statistical Monte Carlo ─────────────────────────────
    console.print(f"\n[bold cyan]Step 3/6:[/] Running Monte Carlo simulation (statistical) …  "
                  f"[dim]{_timestamp()}[/]")
    from src.simulator import TournamentSimulator

    n_sims = settings.get("monte_carlo_iterations", 10_000)
    stat_sim = TournamentSimulator(
        predict_fn=lambda a, b: dc.predict_match(a, b, neutral=True),
        n_simulations=n_sims,
    )
    p_stat = stat_sim.simulate()
    top_stat = sorted(p_stat.items(), key=lambda kv: kv[1], reverse=True)[:3]
    console.print(f"  ✓ {n_sims:,} simulations complete — "
                  f"top 3: {', '.join(f'{t} ({p*100:.1f}%)' for t, p in top_stat)}")

    # ── Step 2b: Match-by-Match Export ──────────────────────────────
    console.print(f"\n[bold cyan]Step 2b/6:[/] Generating Match-by-Match JSON …  [dim]{_timestamp()}[/]")
    from src.simulator import GROUPS
    from src.match_probs import generate_match_by_match_json
    generate_match_by_match_json(dc, GROUPS)
    console.print("  ✓ Match-by-Match probabilities exported to data/processed/match_by_match.json")

    # ── Step 2c: Advanced Props ──────────────────────────────────────
    console.print(f"\n[bold cyan]Step 2c/6:[/] Generating Advanced Props (Stochastic Engine) …  [dim]{_timestamp()}[/]")
    from src.props_model import calculate_team_props, calculate_anytime_goalscorer
    import json
    import math
    
    # Calculate for a sample marquee match: Argentina vs France
    home_team = "Argentina"
    away_team = "France"
    
    # Extract lambda and mu from Dixon-Coles
    att_h, def_h = dc._get_params(home_team)
    att_a, def_a = dc._get_params(away_team)
    ha = 0.0 # Neutral ground
    variance_factor = 2.0
    lambda_val = min(math.exp((att_h + def_a + ha) / variance_factor), 15.0)
    mu_val = min(math.exp((att_a + def_h) / variance_factor), 15.0)
    
    # Baseline historical averages
    arg_corners_avg = 6.0
    arg_cards_avg = 2.0
    fra_corners_avg = 5.0
    fra_cards_avg = 1.5
    
    props_output = {
        "match": f"{home_team} vs {away_team}",
        "team_props": {
            home_team: {
                "over_under_corners_4.5": calculate_team_props(arg_corners_avg, line=4.5),
                "over_under_cards_1.5": calculate_team_props(arg_cards_avg, line=1.5)
            },
            away_team: {
                "over_under_corners_4.5": calculate_team_props(fra_corners_avg, line=4.5),
                "over_under_cards_1.5": calculate_team_props(fra_cards_avg, line=1.5)
            }
        },
        "player_props": {
            "anytime_goalscorer": {
                "Lionel Messi (ARG)": calculate_anytime_goalscorer(lambda_val, player_open_play_share=0.35, is_penalty_taker=True),
                "Ángel Correa (ARG)": calculate_anytime_goalscorer(lambda_val, player_open_play_share=0.10, is_penalty_taker=False),
                "Kylian Mbappé (FRA)": calculate_anytime_goalscorer(mu_val, player_open_play_share=0.40, is_penalty_taker=True)
            }
        }
    }
    
    with open("data/processed/props_probabilities.json", "w", encoding="utf-8") as f:
        json.dump(props_output, f, indent=2)
        
    console.print("  ✓ Advanced props exported to data/processed/props_probabilities.json")

    # ── Step 3: Macro Enablers ──────────────────────────────────────
    console.print(f"\n[bold cyan]Step 3.5/6:[/] Fetching Macro Data (GDP, Population, Climate) …  "
                  f"[dim]{_timestamp()}[/]")
    from src.fetch_macro_data import build_macro_dataset

    macro_df = build_macro_dataset()
    console.print(f"  ✓ Macro data ready with [bold]{len(macro_df)}[/] records")

    # ── Step 4: ML models ───────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 4/6:[/] Training ML models …  [dim]{_timestamp()}[/]")
    from src.ml_engine.catboost_model import CatBoostPredictor
    from src.ml_engine.mcmc_model import MCMCPredictor

    dc_params = {"attack": dc.attack, "defence": dc.defence}

    catboost_pred = CatBoostPredictor(settings)
    cb_metrics = catboost_pred.fit(matches, rankings, dc_params, macro_df)
    console.print(f"  ✓ CatBoost trained — "
                  f"best_iter={cb_metrics.get('best_iteration', 'N/A')}")

    mcmc_pred = MCMCPredictor(settings)
    mcmc_metrics = mcmc_pred.fit(matches, rankings, dc_params)
    mcmc_acc = mcmc_metrics.get("accuracy")
    mcmc_acc_str = f"{mcmc_acc:.3f}" if isinstance(mcmc_acc, (int, float)) else "N/A"
    console.print(f"  ✓ MCMC trained — accuracy={mcmc_acc_str}  ess={mcmc_metrics.get('ess', 0):.0f}")

    # ── Step 4b: Precompute ML Look-up Table ──────────────────────────
    console.print(f"\n[bold cyan]Step 4b/6:[/] Precomputing ML probability look-up table …  "
                  f"[dim]{_timestamp()}[/]")
    from itertools import combinations
    from src.data_loader import get_wc_teams

    teams = get_wc_teams()
    precomputed_ml_probs: dict[tuple[str, str], tuple[float, float, float]] = {}

    for team_a, team_b in combinations(teams, 2):
        p1 = catboost_pred.predict(team_a, team_b, rankings, dc_params, macro_df=macro_df, neutral=True)
        p2 = mcmc_pred.predict(team_a, team_b, rankings, dc_params, neutral=True)
        avg = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
        precomputed_ml_probs[(team_a, team_b)] = avg
        # Store reverse matchup with flipped win/loss (draw stays the same)
        precomputed_ml_probs[(team_b, team_a)] = (avg[2], avg[1], avg[0])

    console.print(f"  ✓ Precomputed [bold]{len(precomputed_ml_probs):,}[/] matchup probabilities")

    def ml_predict_lookup(home: str, away: str) -> tuple[float, float, float]:
        return precomputed_ml_probs[(home, away)]

    console.print(f"\n[bold cyan]Step 4c/6:[/] Running Monte Carlo simulation (ML) …  "
                  f"[dim]{_timestamp()}[/]")
    ml_sim = TournamentSimulator(
        predict_fn=ml_predict_lookup,
        n_simulations=n_sims,
    )
    p_ml = ml_sim.simulate()
    top_ml = sorted(p_ml.items(), key=lambda kv: kv[1], reverse=True)[:3]
    console.print(f"  ✓ {n_sims:,} simulations complete — "
                  f"top 3: {', '.join(f'{t} ({p*100:.1f}%)' for t, p in top_ml)}")

    # ── Step 5: Market data ─────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 5/6:[/] Fetching Polymarket data …  [dim]{_timestamp()}[/]")
    from src.market_fetcher import MarketFetcher

    token_map: dict[str, str] = {}
    try:
        fetcher = MarketFetcher(settings)
        p_market = fetcher.fetch_market_data()
        token_map = getattr(fetcher, "token_map", {})
        console.print(f"  ✓ Fetched prices for [bold]{len(p_market)}[/] teams")
    except Exception as exc:
        console.print(f"  [bold red]✗ Polymarket API unavailable:[/] {exc}")
        console.print("  [yellow]↳ Falling back to model-only analysis (no market edge detection).[/]")
        # Build a dummy p_market from consensus so the rest of the pipeline
        # can proceed (all edges will be zero → no opportunities).
        p_market = {}

    # ── Step 6: Arbitrage ───────────────────────────────────────────
    console.print(f"\n[bold cyan]Step 6/6:[/] Computing arbitrage opportunities …  "
                  f"[dim]{_timestamp()}[/]")
    from src.arbitrage_engine import ArbitrageEngine

    engine = ArbitrageEngine(settings, bankroll)
    p_consensus, opportunities = engine.analyze(p_stat, p_ml, p_market)

    # ── Output ──────────────────────────────────────────────────────
    from src.dashboard import render_dashboard, export_json, append_log

    render_dashboard(
        p_consensus, p_stat, p_ml, p_market,
        opportunities, bankroll,
        console=console,
    )

    # Ensure output directories exist and write results
    try:
        export_json(
            p_consensus, opportunities,
            p_stat=p_stat, p_ml=p_ml, p_market=p_market,
        )
        console.print("[dim]  JSON snapshot saved to data/processed/opportunities.json[/]")
    except OSError as exc:
        console.print(f"[red]  ✗ JSON export failed: {exc}[/]")

    try:
        append_log(opportunities)
        console.print("[dim]  CSV log appended to data/processed/arbitrage_log.csv[/]")
    except OSError as exc:
        console.print(f"[red]  ✗ CSV log failed: {exc}[/]")

    # ── Step 7: Trade Execution Preview ─────────────────────────────
    if opportunities:
        from src.trade_executor import TradeExecutor
        executor = TradeExecutor(settings, token_map=token_map)
        orders = executor.prepare_orders(opportunities)
        executor.preview_orders(orders, console=console)
        try:
            executor.export_orders(orders)
            console.print("[dim]  Trade instructions exported to data/processed/trade_instructions.json[/]")
        except Exception as exc:
            console.print(f"[red]  ✗ Trade export failed: {exc}[/]")

    return len(opportunities)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the pipeline."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="World Cup 2026 Hybrid Statistical Arbitrage System",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single iteration (no polling loop).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview trades without executing.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute trades against Polymarket CLOB (requires valid API key).",
    )
    args = parser.parse_args()

    console = Console()
    settings = _load_settings()

    # Display startup banner
    bankroll = float(os.getenv("BANKROLL", "10000"))
    kelly_pct = settings.get("kelly_fraction", 0.25) * 100
    ev_pct = settings.get("ev_threshold", 0.02) * 100
    interval_min = settings.get("polling_interval_minutes", 60)

    console.print()
    console.print(
        Panel.fit(
            "[bold white]⚽ World Cup 2026 — Hybrid Statistical Arbitrage System[/]\n"
            f"[dim]Bankroll: ${bankroll:,.2f}  |  "
            f"Kelly: {kelly_pct:.0f}%  |  "
            f"EV Threshold: {ev_pct:.0f}%  |  "
            f"Mode: {'single-run' if args.once else f'poll every {interval_min}min'}[/]",
            border_style="bright_cyan",
        )
    )
    console.print()

    if args.once:
        run_pipeline(settings, console)
        return

    # Continuous polling loop
    interval_sec = interval_min * 60
    iteration = 0
    while True:
        iteration += 1
        try:
            console.rule(f"[bold cyan]Iteration {iteration}  •  {_timestamp()}[/]")
            n_opps = run_pipeline(settings, console)
            console.print(
                f"\n[dim]Pipeline complete — {n_opps} opportunity(ies). "
                f"Next run in {interval_min} minutes … (Ctrl-C to quit)[/]"
            )
            time.sleep(interval_sec)

        except KeyboardInterrupt:
            console.print("\n[yellow]⏹  Shutting down gracefully …[/]")
            break

        except Exception:
            logger.exception("Pipeline iteration %d failed.", iteration)
            console.print(
                "[red bold]Pipeline error — retrying in 60 s. "
                "See logs for details.[/]"
            )
            time.sleep(60)


if __name__ == "__main__":
    main()
