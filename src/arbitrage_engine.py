"""Arbitrage engine for World Cup 2026 statistical arbitrage system.

Combines statistical (Dixon-Coles simulation) and ML model outputs into
a consensus probability distribution, then identifies positive expected-value
betting opportunities against Polymarket implied probabilities.  Position
sizing uses fractional Kelly criterion with configurable guardrails.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Opportunity:
    """A single positive-EV betting opportunity."""

    team: str
    p_consensus: float   # Weighted consensus probability
    p_market: float      # Polymarket implied probability
    p_stat: float        # Statistical model probability
    p_ml: float          # ML model probability
    odds: float          # Decimal odds (1 / p_market)
    ev: float            # Expected value as a decimal (0.05 → 5 %)
    kelly_full: float    # Full Kelly fraction
    kelly_fraction: float  # Fractional Kelly (quarter-Kelly by default)
    position_size: float  # Dollar amount to bet
    edge: float           # p_consensus − p_market

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation suitable for JSON export."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ArbitrageEngine:
    """Combine model outputs and identify betting opportunities.

    Parameters
    ----------
    settings : dict
        Full application settings dict (must contain ``consensus_weights``,
        ``kelly_fraction``, ``max_position_pct``, ``ev_threshold``).
    bankroll : float
        Current bankroll in USD.
    """

    def __init__(self, settings: dict, bankroll: float) -> None:
        weights = settings.get("consensus_weights", {})
        self.w_stat: float = float(weights.get("statistical", 0.5))
        self.w_ml: float = float(weights.get("ml", 0.5))

        self.kelly_fraction_multiplier: float = float(
            settings.get("kelly_fraction", 0.25)
        )
        self.max_position_pct: float = float(
            settings.get("max_position_pct", 0.05)
        )
        self.ev_threshold: float = float(
            settings.get("ev_threshold", 0.02)
        )
        self.bankroll: float = float(bankroll)

        logger.info(
            "ArbitrageEngine initialised — w_stat=%.2f  w_ml=%.2f  "
            "kelly=%.2f  max_pos=%.2f  ev_thresh=%.2f  bankroll=$%s",
            self.w_stat,
            self.w_ml,
            self.kelly_fraction_multiplier,
            self.max_position_pct,
            self.ev_threshold,
            f"{self.bankroll:,.2f}",
        )

    # ------------------------------------------------------------------
    # Consensus computation
    # ------------------------------------------------------------------

    def compute_consensus(
        self,
        p_stat: dict[str, float],
        p_ml: dict[str, float],
    ) -> dict[str, float]:
        """Compute a normalised, weighted consensus probability distribution.

        Only teams present in *both* dictionaries are included.  The raw
        weighted averages are then re-normalised so that the distribution
        sums to 1.0.

        Parameters
        ----------
        p_stat : dict[str, float]
            Statistical-model win probabilities (team → probability).
        p_ml : dict[str, float]
            ML-model win probabilities (team → probability).

        Returns
        -------
        dict[str, float]
            Normalised consensus probabilities.
        """
        common_teams = set(p_stat) & set(p_ml)
        if not common_teams:
            logger.warning("No common teams between stat and ML dicts.")
            return {}

        raw: dict[str, float] = {}
        for team in common_teams:
            raw[team] = self.w_stat * p_stat[team] + self.w_ml * p_ml[team]

        total = sum(raw.values())
        if total <= 0:
            logger.warning("Consensus total is non-positive (%.6f).", total)
            return {t: 0.0 for t in raw}

        consensus = {team: prob / total for team, prob in raw.items()}

        logger.info(
            "Consensus computed for %d teams (sum=%.6f).",
            len(consensus),
            sum(consensus.values()),
        )
        return consensus

    # ------------------------------------------------------------------
    # Opportunity detection
    # ------------------------------------------------------------------

    def find_opportunities(
        self,
        p_consensus: dict[str, float],
        p_market: dict[str, float],
        p_stat: dict[str, float] | None = None,
        p_ml: dict[str, float] | None = None,
    ) -> list[Opportunity]:
        """Identify all positive-EV opportunities above the threshold.

        Parameters
        ----------
        p_consensus : dict[str, float]
            Consensus model probabilities.
        p_market : dict[str, float]
            Polymarket implied probabilities.
        p_stat : dict[str, float] | None
            Raw statistical probabilities (used only for annotation).
        p_ml : dict[str, float] | None
            Raw ML probabilities (used only for annotation).

        Returns
        -------
        list[Opportunity]
            Opportunities sorted by EV descending.
        """
        p_stat = p_stat or {}
        p_ml = p_ml or {}

        opportunities: list[Opportunity] = []

        for team in p_consensus:
            if team not in p_market:
                continue

            mkt = p_market[team]
            if mkt <= 0:
                logger.debug("Skipping %s — market prob is zero.", team)
                continue

            cons = p_consensus[team]
            odds = 1.0 / mkt

            # Viability Filter (Longshot Bias Protection)
            if odds > 100.0:
                logger.debug("Skipping %s — market odds > 100.0 (implied prob < 1.0%%).", team)
                continue
            if cons < 0.02:
                logger.debug("Skipping %s — consensus prob < 2.0%%.", team)
                continue

            # Expected value: EV = p * (odds − 1) − (1 − p)
            ev = cons * (odds - 1.0) - (1.0 - cons)

            if ev <= self.ev_threshold:
                continue

            # Kelly criterion: f* = (p * odds − 1) / (odds − 1)
            denom = odds - 1.0
            if denom <= 0:
                continue

            kelly_full = (cons * odds - 1.0) / denom
            kelly_frac = kelly_full * self.kelly_fraction_multiplier

            # Cap position at max_position_pct of bankroll
            position_fraction = min(kelly_frac, self.max_position_pct)
            # Kelly can technically go negative in edge cases — clamp to 0.
            position_fraction = max(position_fraction, 0.0)
            position_size = position_fraction * self.bankroll

            edge = cons - mkt

            opp = Opportunity(
                team=team,
                p_consensus=cons,
                p_market=mkt,
                p_stat=p_stat.get(team, 0.0),
                p_ml=p_ml.get(team, 0.0),
                odds=odds,
                ev=ev,
                kelly_full=kelly_full,
                kelly_fraction=kelly_frac,
                position_size=round(position_size, 2),
                edge=edge,
            )
            opportunities.append(opp)

            logger.debug(
                "Opportunity: %s  EV=%.4f  kelly_frac=%.4f  pos=$%.2f",
                team,
                ev,
                kelly_frac,
                position_size,
            )

        # Sort by EV descending
        opportunities.sort(key=lambda o: o.ev, reverse=True)

        logger.info(
            "Found %d opportunities above EV threshold %.2f%%.",
            len(opportunities),
            self.ev_threshold * 100,
        )
        return opportunities

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def analyze(
        self,
        p_stat: dict[str, float],
        p_ml: dict[str, float],
        p_market: dict[str, float],
    ) -> tuple[dict[str, float], list[Opportunity]]:
        """Run the full consensus → opportunity pipeline.

        Parameters
        ----------
        p_stat : dict[str, float]
            Statistical-model win probabilities.
        p_ml : dict[str, float]
            ML-model win probabilities.
        p_market : dict[str, float]
            Market implied probabilities.

        Returns
        -------
        tuple[dict[str, float], list[Opportunity]]
            ``(p_consensus, opportunities)``
        """
        p_consensus = self.compute_consensus(p_stat, p_ml)
        opportunities = self.find_opportunities(
            p_consensus, p_market, p_stat=p_stat, p_ml=p_ml
        )
        return p_consensus, opportunities
