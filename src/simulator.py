"""
Tournament Simulator — World Cup 2026 Statistical Arbitrage System

Monte Carlo simulation of the 2026 FIFA World Cup:
  • 12 groups of 4 (round-robin)
  • 8 best third-place teams advance
  • R32 → R16 → QF → SF → Final
"""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from typing import Callable


# ======================================================================
# Official 2026 FIFA World Cup groups
# ======================================================================

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Türkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Ukraine"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# ======================================================================
# R32 bracket slot definitions
#
# Each slot maps to either:
#   ("1X", None)      — 1st place in group X
#   ("2X", None)      — 2nd place in group X
#   ("3rd", <pool>)   — one of the best third-place teams from <pool>
# ======================================================================

R32_SLOTS: dict[int, tuple[str, str | None]] = {
    49: ("2A", None),     50: ("1C", None),
    51: ("1E", None),     52: ("1F", None),
    53: ("2E", None),     54: ("1I", None),
    55: ("1A", None),     56: ("1L", None),
    57: ("1G", None),     58: ("1D", None),
    59: ("1H", None),     60: ("2K", None),
    61: ("1B", None),     62: ("2D", None),
    63: ("1J", None),     64: ("1K", None),
}

R32_SLOTS_VS: dict[int, tuple[str, str | None]] = {
    49: ("2B", None),     50: ("2F", None),
    51: ("3rd", "ABCDF"), 52: ("2C", None),
    53: ("2I", None),     54: ("3rd", "CDFGH"),
    55: ("3rd", "CEFHI"), 56: ("3rd", "EHIJK"),
    57: ("3rd", "AEHIJ"), 58: ("3rd", "BEFIJ"),
    59: ("2J", None),     60: ("2L", None),
    61: ("3rd", "EFGIJ"), 62: ("2G", None),
    63: ("2H", None),     64: ("3rd", "DEIJL"),
}

# R32 match pairings  (slot_a  vs  slot_b)
R32_MATCHES: list[tuple[int, int]] = [
    (49, 49),  # these are paired with the _VS table
]

# R16 bracket
R16_BRACKET: dict[int, tuple[int, int]] = {
    65: (49, 51),
    66: (50, 53),
    67: (52, 54),
    68: (55, 56),
    69: (59, 60),
    70: (57, 58),
    71: (62, 64),
    72: (61, 63),
}

# QF bracket
QF_BRACKET: dict[int, tuple[int, int]] = {
    73: (65, 66),
    74: (67, 68),
    75: (69, 70),
    76: (71, 72),
}

# SF bracket
SF_BRACKET: dict[int, tuple[int, int]] = {
    77: (73, 74),
    78: (75, 76),
}

# Final
FINAL_BRACKET: tuple[int, int] = (77, 78)


# ======================================================================
# Third-place allocation helpers
# ======================================================================

# Slots that require a third-place team, with the pool of eligible groups.
_THIRD_PLACE_SLOTS: list[tuple[int, str]] = [
    (51, "ABCDF"),
    (54, "CDFGH"),
    (55, "CEFHI"),
    (56, "EHIJK"),
    (57, "AEHIJ"),
    (58, "BEFIJ"),
    (61, "EFGIJ"),
    (64, "DEIJL"),
]


_ALLOCATION_CACHE: dict[tuple[str, ...], dict[int, str]] = {}

def _allocate_third_place_teams(
    qualifying_groups: list[str],
) -> dict[int, str]:
    """
    Given the sorted list of 8 group letters whose 3rd-place teams
    qualified, assign each to an R32 slot using backtracking to satisfy
    pool constraints. Results are cached for speed.
    """
    cache_key = tuple(qualifying_groups)
    if cache_key in _ALLOCATION_CACHE:
        return _ALLOCATION_CACHE[cache_key]

    slots = [s for s, _ in _THIRD_PLACE_SLOTS]
    pools = [p for _, p in _THIRD_PLACE_SLOTS]
    allocation: dict[int, str] = {}

    def backtrack(idx: int, available: list[str]) -> bool:
        if idx == len(slots):
            return True
        slot = slots[idx]
        pool = pools[idx]
        for g in available:
            if g in pool:
                allocation[slot] = g
                new_avail = list(available)
                new_avail.remove(g)
                if backtrack(idx + 1, new_avail):
                    return True
        return False

    if backtrack(0, qualifying_groups):
        _ALLOCATION_CACHE[cache_key] = dict(allocation)
        return allocation

    # Fallback if no perfect matching exists (force assignment)
    allocation.clear()
    remaining = list(qualifying_groups)
    for slot, pool in _THIRD_PLACE_SLOTS:
        assigned = False
        for g in list(remaining):
            if g in pool:
                allocation[slot] = g
                remaining.remove(g)
                assigned = True
                break
        if not assigned and remaining:
            allocation[slot] = remaining.pop(0)

    _ALLOCATION_CACHE[cache_key] = dict(allocation)
    return allocation


# ======================================================================
# Simulator
# ======================================================================


class TournamentSimulator:
    """
    Monte Carlo simulator for the 2026 FIFA World Cup.

    Parameters
    ----------
    predict_fn : Callable[[str, str], tuple[float, float, float]]
        Function that takes ``(team_a, team_b)`` and returns
        ``(p_a_win, p_draw, p_b_win)``.
    n_simulations : int
        Number of tournament iterations.
    """

    def __init__(
        self,
        predict_fn: Callable[[str, str], tuple[float, float, float]],
        n_simulations: int = 10_000,
    ) -> None:
        self.predict_fn = predict_fn
        self.n_simulations = n_simulations

    # ------------------------------------------------------------------
    # Match simulation
    # ------------------------------------------------------------------

    def _simulate_group_match(
        self, team_a: str, team_b: str
    ) -> tuple[int, int]:
        """
        Simulate a group-stage match and return ``(goals_a, goals_b)``.

        Uses simplified scorelines: win → 2-0, draw → 1-1, loss → 0-2.
        """
        p_a, p_draw, p_b = self.predict_fn(team_a, team_b)
        total = p_a + p_draw + p_b
        if total <= 0:
            # Fallback: equal probabilities
            p_a, p_draw, p_b = 1 / 3, 1 / 3, 1 / 3
            total = 1.0

        p_a /= total
        p_draw /= total
        # p_b is implicit

        r = random.random()
        if r < p_a:
            return (2, 0)
        elif r < p_a + p_draw:
            return (1, 1)
        else:
            return (0, 2)

    def _simulate_knockout_match(self, team_a: str, team_b: str) -> str:
        """
        Simulate a knockout match and return the winning team.

        If the model predicts a draw, a 50/50 penalty shoot-out decides.
        """
        p_a, p_draw, p_b = self.predict_fn(team_a, team_b)
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
            # Draw → coin flip (penalty shoot-out)
            return team_a if random.random() < 0.5 else team_b
        else:
            return team_b

    # ------------------------------------------------------------------
    # Group stage
    # ------------------------------------------------------------------

    def _simulate_group(
        self, teams: list[str]
    ) -> tuple[str, str, dict[str, dict[str, int]]]:
        """
        Simulate a round-robin group of 4 teams.

        Returns
        -------
        first : str
            Group winner.
        second : str
            Runner-up.
        records : dict[str, dict]
            ``{team: {"pts": int, "gd": int, "gf": int}}`` for all 4.
        """
        records: dict[str, dict[str, int]] = {
            t: {"pts": 0, "gd": 0, "gf": 0} for t in teams
        }

        for t_a, t_b in combinations(teams, 2):
            ga, gb = self._simulate_group_match(t_a, t_b)
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

        # Rank: points desc → goal difference desc → goals for desc
        ranking = sorted(
            teams,
            key=lambda t: (
                records[t]["pts"],
                records[t]["gd"],
                records[t]["gf"],
            ),
            reverse=True,
        )

        return ranking[0], ranking[1], {t: records[t] for t in ranking}

    # ------------------------------------------------------------------
    # Full tournament simulation
    # ------------------------------------------------------------------

    def _simulate_once(self) -> str:
        """Run one full tournament and return the champion."""

        # ---- Group stage ------------------------------------------------
        group_results: dict[str, tuple[str, str, dict]] = {}
        # Maps: group_letter → (1st, 2nd, full_records)
        third_place_candidates: list[tuple[str, str, dict[str, int]]] = []
        # (group_letter, team_name, record_dict)

        for letter, teams in GROUPS.items():
            first, second, records = self._simulate_group(teams)
            group_results[letter] = (first, second, records)

            # The third-place team is the one ranked 3rd
            ranked = sorted(
                teams,
                key=lambda t: (
                    records[t]["pts"],
                    records[t]["gd"],
                    records[t]["gf"],
                ),
                reverse=True,
            )
            third_team = ranked[2]
            third_place_candidates.append(
                (letter, third_team, records[third_team])
            )

        # ---- Determine best 8 third-place teams ------------------------
        third_place_candidates.sort(
            key=lambda x: (x[2]["pts"], x[2]["gd"], x[2]["gf"]),
            reverse=True,
        )
        qualifying_thirds = third_place_candidates[:8]
        qualifying_groups_sorted = sorted([g for g, _, _ in qualifying_thirds])

        # Map group letter → team name for qualifying 3rds
        third_team_by_group: dict[str, str] = {
            g: t for g, t, _ in qualifying_thirds
        }

        # Allocate to R32 slots
        slot_to_third_group = _allocate_third_place_teams(
            qualifying_groups_sorted
        )

        # ---- Build R32 matchups -----------------------------------------
        def _resolve_slot(
            slot_def: tuple[str, str | None],
            side: str,
        ) -> str:
            """Convert a slot definition to an actual team name."""
            code, pool = slot_def
            if code.startswith("1"):
                grp = code[1]
                return group_results[grp][0]
            elif code.startswith("2"):
                grp = code[1]
                return group_results[grp][1]
            elif code == "3rd":
                # Look up which group was assigned to this slot
                # We need the slot number to look up. Pass it via side.
                raise ValueError("Use _resolve_third")
            else:
                raise ValueError(f"Unknown slot code: {code}")

        def _resolve_team(slot_num: int, table: dict) -> str:
            code, pool = table[slot_num]
            if code == "3rd":
                grp = slot_to_third_group.get(slot_num)
                if grp is None:
                    # Fallback: pick any unallocated qualifying third
                    raise RuntimeError(
                        f"No third-place team allocated to slot {slot_num}"
                    )
                return third_team_by_group[grp]
            elif code.startswith("1"):
                return group_results[code[1]][0]
            elif code.startswith("2"):
                return group_results[code[1]][1]
            else:
                raise ValueError(f"Unknown code: {code}")

        # R32 winners
        r32_winners: dict[int, str] = {}
        for slot_num in R32_SLOTS:
            team_a = _resolve_team(slot_num, R32_SLOTS)
            team_b = _resolve_team(slot_num, R32_SLOTS_VS)
            r32_winners[slot_num] = self._simulate_knockout_match(
                team_a, team_b
            )

        # R16 winners
        r16_winners: dict[int, str] = {}
        for slot_num, (sa, sb) in R16_BRACKET.items():
            r16_winners[slot_num] = self._simulate_knockout_match(
                r32_winners[sa], r32_winners[sb]
            )

        # QF winners
        qf_winners: dict[int, str] = {}
        for slot_num, (sa, sb) in QF_BRACKET.items():
            qf_winners[slot_num] = self._simulate_knockout_match(
                r16_winners[sa], r16_winners[sb]
            )

        # SF winners
        sf_winners: dict[int, str] = {}
        for slot_num, (sa, sb) in SF_BRACKET.items():
            sf_winners[slot_num] = self._simulate_knockout_match(
                qf_winners[sa], qf_winners[sb]
            )

        # Final
        sa, sb = FINAL_BRACKET
        champion = self._simulate_knockout_match(
            sf_winners[sa], sf_winners[sb]
        )

        return champion

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate(self) -> dict[str, float]:
        """
        Run the full Monte Carlo simulation.

        Returns
        -------
        dict[str, float]
            ``{team_name: win_probability}`` for every team that won at
            least one simulation. Sorted descending by probability.
        """
        wins: dict[str, int] = defaultdict(int)

        for i in range(1, self.n_simulations + 1):
            champion = self._simulate_once()
            wins[champion] += 1

            if i % 1000 == 0:
                print(
                    f"[Simulator] {i:,}/{self.n_simulations:,} iterations complete"
                )

        # Convert to probabilities
        probs: dict[str, float] = {
            team: count / self.n_simulations
            for team, count in wins.items()
        }

        # Sort descending
        probs = dict(sorted(probs.items(), key=lambda kv: -kv[1]))

        return probs
