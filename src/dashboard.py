"""Dashboard and reporting module for the World Cup 2026 arbitrage system.

Provides three main outputs:

1. **Rich console dashboard** — colourful probability comparison tables
   and opportunity highlights rendered in the terminal.
2. **JSON export** — structured snapshot written to disk for downstream
   tooling or dashboards.
3. **CSV append log** — running log of every opportunity detected across
   pipeline iterations.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.arbitrage_engine import Opportunity

logger = logging.getLogger(__name__)


# ── Rich Console Dashboard ─────────────────────────────────────────────

def render_dashboard(
    p_consensus: dict[str, float],
    p_stat: dict[str, float],
    p_ml: dict[str, float],
    p_market: dict[str, float],
    opportunities: Sequence[Opportunity],
    bankroll: float,
    *,
    console: Console | None = None,
) -> None:
    """Render the full dashboard to the terminal via ``rich``.

    Parameters
    ----------
    p_consensus : dict[str, float]
        Consensus probabilities.
    p_stat, p_ml, p_market : dict[str, float]
        Component probability dictionaries.
    opportunities : Sequence[Opportunity]
        Detected opportunities (already sorted by EV).
    bankroll : float
        Current bankroll for display.
    console : Console | None
        Optional pre-existing ``rich.Console`` instance.
    """
    if console is None:
        console = Console()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── Header ──────────────────────────────────────────────────────
    console.print()
    console.print(
        Panel.fit(
            f"[bold white]⚽ World Cup 2026 — Hybrid Statistical Arbitrage System[/]\n"
            f"[dim]{now}  |  Bankroll: ${bankroll:,.2f}[/]",
            border_style="bright_cyan",
        )
    )
    console.print()

    # ── Table 1: Full Probability Comparison ────────────────────────
    _render_probability_table(console, p_consensus, p_stat, p_ml, p_market)
    console.print()

    # ── Table 2: Arbitrage Opportunities ────────────────────────────
    _render_opportunities_table(console, opportunities)
    console.print()

    # ── Summary panel ───────────────────────────────────────────────
    _render_summary_panel(console, opportunities, bankroll)
    console.print()


def _render_probability_table(
    console: Console,
    p_consensus: dict[str, float],
    p_stat: dict[str, float],
    p_ml: dict[str, float],
    p_market: dict[str, float],
) -> None:
    """Render the full 48-team probability comparison table."""
    table = Table(
        title="[bold]Full Probability Comparison[/]",
        title_style="bright_cyan",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Rank", style="dim", justify="right", width=4)
    table.add_column("Team", style="bold white", min_width=20)
    table.add_column("P_consensus", justify="right")
    table.add_column("P_stat", justify="right")
    table.add_column("P_ml", justify="right")
    table.add_column("P_market", justify="right")
    table.add_column("Edge", justify="right")

    # Sort teams by consensus probability descending
    teams_sorted = sorted(
        p_consensus.keys(),
        key=lambda t: p_consensus.get(t, 0.0),
        reverse=True,
    )

    for rank, team in enumerate(teams_sorted, start=1):
        cons = p_consensus.get(team, 0.0)
        stat = p_stat.get(team, 0.0)
        ml = p_ml.get(team, 0.0)
        mkt = p_market.get(team, 0.0)
        edge = cons - mkt

        # Colour-code edge
        if edge > 0.005:
            edge_text = Text(f"+{edge * 100:.2f}%", style="bold green")
        elif edge < -0.005:
            edge_text = Text(f"{edge * 100:.2f}%", style="bold red")
        else:
            edge_text = Text(f"{edge * 100:.2f}%", style="dim")

        table.add_row(
            str(rank),
            team,
            f"{cons * 100:.2f}%",
            f"{stat * 100:.2f}%",
            f"{ml * 100:.2f}%",
            f"{mkt * 100:.2f}%" if mkt else "—",
            edge_text,
        )

    console.print(table)


def _render_opportunities_table(
    console: Console,
    opportunities: Sequence[Opportunity],
) -> None:
    """Render the arbitrage opportunities table."""
    table = Table(
        title="[bold]Arbitrage Opportunities (EV > 2%)[/]",
        title_style="bright_yellow",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Team", style="bold white", min_width=20)
    table.add_column("P_consensus", justify="right")
    table.add_column("P_market", justify="right")
    table.add_column("Odds", justify="right")
    table.add_column("EV%", justify="right", style="bold green")
    table.add_column("Kelly%", justify="right")
    table.add_column("Position ($)", justify="right", style="bold cyan")

    if not opportunities:
        console.print(
            Panel(
                "[dim italic]No opportunities with EV > 2% detected. "
                "Market appears efficient.[/]",
                border_style="yellow",
            )
        )
        return

    for opp in opportunities:
        table.add_row(
            opp.team,
            f"{opp.p_consensus * 100:.2f}%",
            f"{opp.p_market * 100:.2f}%",
            f"{opp.odds:.2f}",
            f"{opp.ev * 100:.2f}%",
            f"{opp.kelly_fraction * 100:.2f}%",
            f"${opp.position_size:,.2f}",
        )

    console.print(table)


def _render_summary_panel(
    console: Console,
    opportunities: Sequence[Opportunity],
    bankroll: float,
) -> None:
    """Render the summary statistics panel."""
    count = len(opportunities)
    total_position = sum(o.position_size for o in opportunities)
    avg_ev = (
        sum(o.ev for o in opportunities) / count if count else 0.0
    )
    pct_deployed = (total_position / bankroll * 100) if bankroll else 0.0

    console.print(
        Panel.fit(
            f"[bold]Summary[/]\n"
            f"  Opportunities found : [bold cyan]{count}[/]\n"
            f"  Capital deployed    : [bold cyan]${total_position:,.2f}[/]  "
            f"({pct_deployed:.1f}% of bankroll)\n"
            f"  Average EV          : [bold green]{avg_ev * 100:.2f}%[/]",
            border_style="bright_green",
        )
    )


# ── JSON Export ────────────────────────────────────────────────────────

def export_json(
    p_consensus: dict[str, float],
    opportunities: Sequence[Opportunity],
    output_path: str = "data/processed/opportunities.json",
    *,
    p_stat: dict[str, float] | None = None,
    p_ml: dict[str, float] | None = None,
    p_market: dict[str, float] | None = None,
) -> Path:
    """Write a structured JSON snapshot of probabilities and opportunities.

    Parameters
    ----------
    p_consensus : dict[str, float]
        Consensus probabilities.
    opportunities : Sequence[Opportunity]
        List of detected opportunities.
    output_path : str
        Destination file path (relative or absolute).
    p_stat, p_ml, p_market : dict[str, float] | None
        Optional component probabilities included in the export.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    p_stat = p_stat or {}
    p_ml = p_ml or {}
    p_market = p_market or {}

    now = datetime.now(timezone.utc)
    
    dest = Path(output_path)
    existing_history = {}
    if dest.exists():
        try:
            with dest.open("r", encoding="utf-8") as f:
                old_data = json.load(f)
                for tm in old_data.get("tracked_markets", []):
                    existing_history[tm["team"]] = tm.get("ev_history", [])
        except Exception:
            pass

    # Build per-team probability breakdown
    all_probabilities: dict[str, dict[str, float]] = {}
    tracked_markets = []
    
    for team in sorted(p_consensus):
        pc = p_consensus.get(team, 0.0)
        pm = p_market.get(team, 0.0)
        
        all_probabilities[team] = {
            "p_consensus": round(pc, 6),
            "p_stat": round(p_stat.get(team, 0.0), 6),
            "p_ml": round(p_ml.get(team, 0.0), 6),
            "p_market": round(pm, 6),
        }
        
        odds = (1.0 / pm) if pm > 0 else 0.0
        ev = (pc * odds) - 1.0 if pm > 0 else 0.0
        b = odds - 1.0
        kelly = (ev / b) if b > 0 else 0.0
        kelly = max(0.0, kelly)
        
        ev_pct = round(ev * 100, 2)
        hist = existing_history.get(team, [])
        hist.append(ev_pct)
        # Keep last 10 points for sparkline
        hist = hist[-10:]
        
        tracked_markets.append({
            "team": team,
            "p_consensus": round(pc, 6),
            "p_market": round(pm, 6),
            "odds": round(odds, 2),
            "ev_percent": ev_pct,
            "kelly_fraction": round(kelly * 100, 2),
            "ev_history": hist
        })
        
    # Sort tracked_markets by consensus probability
    tracked_markets.sort(key=lambda x: x['p_consensus'], reverse=True)

    count = len(opportunities)
    total_position = sum(o.position_size for o in opportunities)
    avg_ev = sum(o.ev for o in opportunities) / count if count else 0.0

    payload: dict[str, Any] = {
        "timestamp": now.isoformat(),
        "all_probabilities": all_probabilities,
        "tracked_markets": tracked_markets,
        "opportunities": [o.to_dict() for o in opportunities],
        "summary": {
            "count": count,
            "total_position": round(total_position, 2),
            "avg_ev": round(avg_ev, 6),
        },
    }

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("JSON exported to %s (%d teams, %d opportunities).", dest, len(all_probabilities), count)
    return dest


# ── CSV Append Log ─────────────────────────────────────────────────────

_CSV_COLUMNS = [
    "timestamp",
    "team",
    "p_consensus",
    "p_market",
    "ev",
    "kelly_fraction",
    "position_size",
]


def append_log(
    opportunities: Sequence[Opportunity],
    log_path: str = "data/processed/arbitrage_log.csv",
) -> Path:
    """Append detected opportunities to a rolling CSV log.

    Creates the file (with header) if it does not yet exist.

    Parameters
    ----------
    opportunities : Sequence[Opportunity]
        Opportunities to log.
    log_path : str
        CSV file path (relative or absolute).

    Returns
    -------
    Path
        Resolved path to the log file.
    """
    dest = Path(log_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    file_exists = dest.exists() and dest.stat().st_size > 0

    now = datetime.now(timezone.utc).isoformat()

    with dest.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for opp in opportunities:
            writer.writerow(
                {
                    "timestamp": now,
                    "team": opp.team,
                    "p_consensus": round(opp.p_consensus, 6),
                    "p_market": round(opp.p_market, 6),
                    "ev": round(opp.ev, 6),
                    "kelly_fraction": round(opp.kelly_fraction, 6),
                    "position_size": round(opp.position_size, 2),
                }
            )

    logger.info(
        "Appended %d rows to %s.", len(opportunities), dest
    )
    return dest
