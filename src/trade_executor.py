"""Trade Execution Preview — DRY-RUN ONLY.

Generates structured trade instructions from identified arbitrage
opportunities and renders a professional preview table for manual
execution on the Polymarket UI.

**This module is strictly advisory.** It does NOT submit orders, sign
transactions, handle private keys, or interact with any exchange API.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.arbitrage_engine import Opportunity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DISCLAIMER = (
    "[!] ADVISORY ONLY -- Execute trades manually on Polymarket UI\n"
    "This output is for informational purposes only. No orders have been "
    "placed, signed, or submitted. Verify all parameters before acting."
)

_HIGH_EV_THRESHOLD = 1.0    # 100 %
_MODERATE_EV_THRESHOLD = 0.5  # 50 %


# ---------------------------------------------------------------------------
# TradeExecutor
# ---------------------------------------------------------------------------

class TradeExecutor:
    """Build, preview, and export trade instructions from opportunities.

    Parameters
    ----------
    settings : dict
        Full application settings dictionary (must contain ``bankroll``).
    token_map : dict[str, str] | None
        Optional mapping of ``team_name`` → CLOB token ID, typically
        populated by :class:`MarketFetcher`.
    """

    def __init__(
        self,
        settings: dict,
        token_map: dict[str, str] | None = None,
    ) -> None:
        self.settings = settings
        self.bankroll: float = float(settings.get("bankroll", 0.0))
        self.token_map: dict[str, str] = token_map or {}

        logger.info(
            "TradeExecutor initialised — bankroll=$%s  token_map_size=%d  "
            "[DRY-RUN MODE]",
            f"{self.bankroll:,.2f}",
            len(self.token_map),
        )

    # ------------------------------------------------------------------
    # Order preparation
    # ------------------------------------------------------------------

    def prepare_orders(
        self, opportunities: list[Opportunity]
    ) -> list[dict[str, Any]]:
        """Convert a list of :class:`Opportunity` objects into order dicts.

        Each order contains the fields required for manual execution on
        Polymarket, plus the analytical context that motivated the trade.

        Parameters
        ----------
        opportunities : list[Opportunity]
            Positive-EV opportunities identified by the arbitrage engine.

        Returns
        -------
        list[dict]
            Order dicts sorted by expected value (descending).
        """
        if not opportunities:
            logger.warning("No opportunities supplied — returning empty order list.")
            return []

        orders: list[dict[str, Any]] = []

        for opp in opportunities:
            limit_price = round(opp.p_market, 4)
            size_usd = round(opp.position_size, 2)
            quantity = round(size_usd / limit_price, 4) if limit_price > 0 else 0.0

            order: dict[str, Any] = {
                "team": opp.team,
                "token_id": self.token_map.get(opp.team),
                "side": "BUY",
                "type": "LIMIT",
                "size_usd": size_usd,
                "limit_price": limit_price,
                "quantity": round(quantity, 4),
                "ev_pct": round(opp.ev * 100, 2),
                "p_consensus": round(opp.p_consensus, 6),
                "p_market": round(opp.p_market, 6),
                "kelly_fraction": round(opp.kelly_fraction * 100, 2),
                "edge": round(opp.edge, 6),
            }
            orders.append(order)

            logger.debug(
                "Prepared order: %s  side=%s  size=$%.2f  limit=%.4f  qty=%.4f",
                opp.team,
                order["side"],
                size_usd,
                limit_price,
                quantity,
            )

        # Sort by EV descending
        orders.sort(key=lambda o: o["ev_pct"], reverse=True)

        logger.info("Prepared %d orders (total capital $%s).",
                     len(orders),
                     f"{sum(o['size_usd'] for o in orders):,.2f}")
        return orders

    # ------------------------------------------------------------------
    # Rich preview
    # ------------------------------------------------------------------

    def preview_orders(
        self,
        orders: list[dict[str, Any]],
        console: Console | None = None,
    ) -> None:
        """Render a professional Rich table of pending trade instructions.

        Parameters
        ----------
        orders : list[dict]
            Order dicts produced by :meth:`prepare_orders`.
        console : Console | None
            Optional Rich console; a new one is created if omitted.
        """
        console = console or Console()

        # -- Disclaimer panel -------------------------------------------
        disclaimer_text = Text(_DISCLAIMER, style="bold yellow")
        panel = Panel(
            disclaimer_text,
            title="[bold red]<<< DRY-RUN MODE >>>[/bold red]",
            title_align="center",
            border_style="red",
            padding=(1, 2),
        )
        console.print()
        console.print(panel)
        console.print()

        if not orders:
            console.print(
                "[dim italic]No trade instructions to display.[/dim italic]"
            )
            return

        # -- Build table ------------------------------------------------
        table = Table(
            title="Trade Execution Preview",
            title_style="bold bright_white",
            show_header=True,
            header_style="bold cyan",
            border_style="bright_black",
            show_lines=True,
            pad_edge=True,
            expand=True,
        )

        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Team", style="bold white", min_width=18)
        table.add_column("Side", justify="center", width=6)
        table.add_column("Type", justify="center", width=7)
        table.add_column("Size ($)", justify="right", min_width=10)
        table.add_column("Limit Price", justify="right", min_width=11)
        table.add_column("Qty", justify="right", min_width=9)
        table.add_column("EV%", justify="right", min_width=8)
        table.add_column("Edge%", justify="right", min_width=8)
        table.add_column("Token ID", style="dim", min_width=14, max_width=20)

        for idx, order in enumerate(orders, start=1):
            ev_pct = order["ev_pct"]

            # Colour-code by EV
            if ev_pct > _HIGH_EV_THRESHOLD * 100:
                row_style = "bold green"
                ev_style = "bold bright_green"
            elif ev_pct > _MODERATE_EV_THRESHOLD * 100:
                row_style = "yellow"
                ev_style = "bold yellow"
            else:
                row_style = "white"
                ev_style = "white"

            token_display = order.get("token_id") or "N/A"
            if token_display != "N/A" and len(token_display) > 18:
                token_display = token_display[:8] + "…" + token_display[-8:]

            table.add_row(
                str(idx),
                order["team"],
                Text(order["side"], style="bold bright_green"),
                order["type"],
                f"${order['size_usd']:,.2f}",
                f"${order['limit_price']:.4f}",
                f"{order['quantity']:,.2f}",
                Text(f"{ev_pct:+.1f}%", style=ev_style),
                Text(f"{order['edge'] * 100:+.1f}%", style=row_style),
                token_display,
            )

        console.print(table)

        # -- Summary footer ---------------------------------------------
        total_capital = sum(o["size_usd"] for o in orders)
        avg_ev = sum(o["ev_pct"] for o in orders) / len(orders)
        max_ev_order = max(orders, key=lambda o: o["ev_pct"])

        summary_lines = (
            f"[bold]Total Orders:[/bold]  {len(orders)}    "
            f"[bold]Capital:[/bold]  ${total_capital:,.2f}    "
            f"[bold]Avg EV:[/bold]  {avg_ev:+.1f}%    "
            f"[bold]Best:[/bold]  {max_ev_order['team']} ({max_ev_order['ev_pct']:+.1f}%)"
        )

        console.print()
        console.print(
            Panel(
                summary_lines,
                title="[bold]Summary[/bold]",
                border_style="bright_blue",
                padding=(0, 2),
            )
        )
        console.print()

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_orders(
        self,
        orders: list[dict[str, Any]],
        path: str = "data/processed/trade_instructions.json",
    ) -> None:
        """Write trade instructions to a JSON file.

        The output includes metadata (timestamp, bankroll, counts) alongside
        the full order list so that downstream tools have context.

        Parameters
        ----------
        orders : list[dict]
            Order dicts from :meth:`prepare_orders`.
        path : str
            Destination file path (parent directories created as needed).
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        total_capital = sum(o["size_usd"] for o in orders)

        payload: dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "bankroll": self.bankroll,
                "total_orders": len(orders),
                "total_capital": round(total_capital, 2),
                "mode": "DRY-RUN / ADVISORY ONLY",
            },
            "orders": orders,
        }

        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Exported %d trade instructions to %s (capital=$%s).",
            len(orders),
            output_path,
            f"{total_capital:,.2f}",
        )

    # ------------------------------------------------------------------
    # Execution summary
    # ------------------------------------------------------------------

    def generate_execution_summary(
        self, orders: list[dict[str, Any]]
    ) -> str:
        """Return a human-readable multi-line execution summary.

        Parameters
        ----------
        orders : list[dict]
            Order dicts from :meth:`prepare_orders`.

        Returns
        -------
        str
            Formatted summary string.
        """
        if not orders:
            return "No trade instructions to summarise."

        total_capital = sum(o["size_usd"] for o in orders)
        best = max(orders, key=lambda o: o["ev_pct"])

        # Risk distribution: capital per team
        capital_by_team: dict[str, float] = {}
        for o in orders:
            capital_by_team[o["team"]] = (
                capital_by_team.get(o["team"], 0.0) + o["size_usd"]
            )

        distribution_lines: list[str] = []
        for team, cap in sorted(
            capital_by_team.items(), key=lambda x: x[1], reverse=True
        ):
            pct_of_bankroll = (cap / self.bankroll * 100) if self.bankroll > 0 else 0.0
            distribution_lines.append(
                f"    {team:<25s}  ${cap:>10,.2f}  ({pct_of_bankroll:.1f}% of bankroll)"
            )

        summary = (
            "=======================================================\n"
            "  TRADE EXECUTION SUMMARY  (DRY-RUN / ADVISORY ONLY)\n"
            "=======================================================\n"
            f"\n"
            f"  Orders to execute:      {len(orders)}\n"
            f"  Total capital to deploy: ${total_capital:,.2f}\n"
            f"  Bankroll:                ${self.bankroll:,.2f}\n"
            f"  Capital utilisation:     "
            f"{(total_capital / self.bankroll * 100) if self.bankroll > 0 else 0:.1f}%\n"
            f"\n"
            f"  Highest EV opportunity:  {best['team']}  "
            f"(EV {best['ev_pct']:+.1f}%  |  Edge {best['edge'] * 100:+.1f}%)\n"
            f"\n"
            f"  Risk Distribution:\n"
            + "\n".join(distribution_lines)
            + "\n\n"
            "=======================================================\n"
            "  [!] Execute trades manually on Polymarket UI\n"
            "======================================================="
        )

        logger.info("Generated execution summary for %d orders.", len(orders))
        return summary
