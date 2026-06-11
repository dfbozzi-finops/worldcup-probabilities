"""
market_fetcher.py — Fetch odds from Polymarket's Gamma API for the
2026 FIFA World Cup outright-winner market.

Reads from:
  • Gamma API   (event + market metadata, NO auth required)
  • CLOB API    (order-book depth for midpoint pricing)

Exports:
  MarketFetcher – single public class consumed by the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Team-name normalisation
# ---------------------------------------------------------------------------
# Prefer the canonical implementation from data_loader when it exists;
# fall back to a lightweight local version so this module is self-contained
# during early development.
# ---------------------------------------------------------------------------
try:
    from src.data_loader import normalize_team_name  # noqa: F401
except ImportError:  # pragma: no cover – data_loader not yet created

    _TEAM_ALIASES: dict[str, str] = {
        "USA": "United States",
        "US": "United States",
        "Korea Republic": "South Korea",
        "Korea DPR": "North Korea",
        "IR Iran": "Iran",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "Türkiye": "Turkey",
        "Turkiye": "Turkey",
        "Czechia": "Czech Republic",
        "China PR": "China",
        "Bosnia and Herzegovina": "Bosnia-Herzegovina",
        "Trinidad and Tobago": "Trinidad & Tobago",
    }

    def normalize_team_name(raw: str) -> str:
        """Trim whitespace and resolve common aliases."""
        name = raw.strip()
        return _TEAM_ALIASES.get(name, name)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
load_dotenv()

logger = logging.getLogger(__name__)

_DEFAULT_SLUG = "2026-fifa-world-cup-winner-595"
_DEFAULT_GAMMA_BASE = "https://gamma-api.polymarket.com"
_DEFAULT_CLOB_BASE = "https://clob.polymarket.com"

# Regex patterns for extracting the team name from market questions
_TEAM_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Will\s+(.+?)\s+win\s+the\s+2026\s+FIFA\s+World\s+Cup", re.IGNORECASE),
    re.compile(r"Will\s+(.+?)\s+win\s+the\s+FIFA\s+World\s+Cup\s+2026", re.IGNORECASE),
    re.compile(r"Will\s+(.+?)\s+win\s+the\s+2026\s+World\s+Cup", re.IGNORECASE),
    re.compile(r"Will\s+(.+?)\s+win\s+the\s+World\s+Cup", re.IGNORECASE),
]

# Retry config
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_FACTOR = 2.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_team_name(question: str) -> str | None:
    """Parse the team name out of a Polymarket question string."""
    for pattern in _TEAM_PATTERNS:
        m = pattern.search(question)
        if m:
            return normalize_team_name(m.group(1))
    return None


def _safe_json_loads(raw: Any, field_name: str = "field") -> Any:
    """Deserialise a stringified JSON value that Gamma API embeds."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            pass
    logger.warning("Failed to parse JSON for %s: %.120s", field_name, str(raw))
    return None


# ---------------------------------------------------------------------------
# MarketFetcher
# ---------------------------------------------------------------------------

class MarketFetcher:
    """Fetches and normalises Polymarket odds for the WC outright market."""

    # ---- construction ---------------------------------------------------- #

    def __init__(self, settings: dict) -> None:
        self.slug: str = settings.get("event_slug", _DEFAULT_SLUG)
        self.gamma_base: str = settings.get("gamma_api_base", _DEFAULT_GAMMA_BASE).rstrip("/")
        self.clob_base: str = settings.get("clob_api_base", _DEFAULT_CLOB_BASE).rstrip("/")

        # API key loaded from env — not required for MVP read-only access
        self.api_key: str | None = os.getenv("POLYMARKET_API_KEY")
        if self.api_key and self.api_key.startswith("your_"):
            # Placeholder value from .env template → treat as unset
            self.api_key = None

        # httpx client with sane defaults
        headers: dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "world-cup-quant/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers=headers,
            follow_redirects=True,
        )

        logger.info(
            "MarketFetcher initialised  slug=%s  gamma=%s  clob=%s  api_key=%s",
            self.slug,
            self.gamma_base,
            self.clob_base,
            "set" if self.api_key else "unset",
        )

    # ---- internal: retrying GET ------------------------------------------ #

    def _get_with_retry(self, url: str, params: dict | None = None) -> httpx.Response:
        """GET *url* with exponential-backoff retry on transient errors."""
        backoff = _INITIAL_BACKOFF_S
        last_exc: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._client.get(url, params=params)
                resp.raise_for_status()
                return resp
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                retryable = True
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    # Client errors (4xx) are not retryable (except 429)
                    if exc.response.status_code != 429:
                        retryable = False
                if not retryable or attempt == _MAX_RETRIES:
                    logger.error(
                        "GET %s failed after %d attempt(s): %s",
                        url, attempt, exc,
                    )
                    raise
                logger.warning(
                    "GET %s attempt %d/%d failed (%s), retrying in %.1fs …",
                    url, attempt, _MAX_RETRIES, exc, backoff,
                )
                time.sleep(backoff)
                backoff *= _BACKOFF_FACTOR

        # Should be unreachable, but keeps mypy happy.
        assert last_exc is not None
        raise last_exc  # pragma: no cover

    # ---- internal: event retrieval --------------------------------------- #

    def _fetch_event(self) -> dict:
        """Return the full event JSON from the Gamma API."""
        url = f"{self.gamma_base}/events/slug/{self.slug}"
        try:
            resp = self._get_with_retry(url)
            return resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                logger.warning("Primary slug '%s' returned 404. Initiating dynamic fallback search...", self.slug)
                return self._fallback_search_event()
            raise

    def _fallback_search_event(self) -> dict:
        """Query Gamma API events endpoint for 'World Cup' to find the active slug."""
        search_url = f"{self.gamma_base}/events"
        try:
            resp = self._get_with_retry(search_url, params={"title": "World Cup"})
            events = resp.json()
            for event in events:
                title = event.get("title", "")
                if "World Cup Winner" in title:
                    new_slug = event.get("slug")
                    if new_slug:
                        logger.info("Fallback search discovered active slug: '%s'", new_slug)
                        self.slug = new_slug
                        return event
            logger.error("Fallback search failed: No 'World Cup Winner' event found.")
        except Exception as exc:
            logger.error("Fallback search encountered an error: %s", exc)
        return {}

    # ---- public: raw prices ---------------------------------------------- #

    def fetch_raw_prices(self) -> dict[str, float]:
        """
        Return ``{team_name: yes_price}`` from the Gamma API.

        Each market in the event is a binary Yes/No for one team.  The
        ``outcomePrices`` field is a *stringified* JSON array like
        ``'["0.12", "0.88"]'``, where index 0 is the Yes price.
        """
        event = self._fetch_event()
        markets: list[dict] = event.get("markets", [])
        if not markets:
            logger.warning("Event '%s' contains no markets", self.slug)
            return {}

        prices: dict[str, float] = {}
        for mkt in markets:
            question: str = mkt.get("question", "")
            team = _extract_team_name(question)
            if team is None:
                logger.debug("Skipping unrecognised question: %s", question)
                continue

            outcome_prices = _safe_json_loads(
                mkt.get("outcomePrices", "[]"), "outcomePrices"
            )
            if not outcome_prices or len(outcome_prices) < 1:
                logger.debug("No outcomePrices for %s", team)
                continue

            try:
                yes_price = float(outcome_prices[0])
            except (ValueError, IndexError):
                logger.warning("Bad yes price for %s: %s", team, outcome_prices)
                continue

            prices[team] = yes_price

        logger.info(
            "Fetched raw prices for %d teams (sample: %s)",
            len(prices),
            dict(list(prices.items())[:5]),
        )
        return prices

    # ---- public: order book ---------------------------------------------- #

    def fetch_order_book(self, token_id: str) -> tuple[float, float]:
        """
        Return ``(best_bid, best_ask)`` for *token_id* from the CLOB API.

        If the order book is empty on either side, the corresponding value
        is returned as ``0.0``.
        """
        url = f"{self.clob_base}/book"
        try:
            resp = self._get_with_retry(url, params={"token_id": token_id})
            book = resp.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as exc:
            logger.warning("Order-book fetch failed for token %s: %s", token_id, exc)
            return (0.0, 0.0)

        bids: list[dict] = book.get("bids", [])
        asks: list[dict] = book.get("asks", [])

        best_bid = float(bids[0]["price"]) if bids else 0.0
        best_ask = float(asks[0]["price"]) if asks else 0.0

        return (best_bid, best_ask)

    # ---- public: midpoint prices ----------------------------------------- #

    def fetch_midpoint_prices(self) -> dict[str, float]:
        """
        Return ``{team_name: midpoint}`` using CLOB order-book data,
        falling back to ``outcomePrices`` when the book is thin.
        """
        event = self._fetch_event()
        markets: list[dict] = event.get("markets", [])
        prices: dict[str, float] = {}

        for mkt in markets:
            question = mkt.get("question", "")
            team = _extract_team_name(question)
            if team is None:
                continue

            # ---- try midpoint from order book ----
            # Locate the "Yes" token ID
            yes_token_id: str | None = None

            # Method 1: parse tokens array
            tokens = mkt.get("tokens", [])
            for tok in tokens:
                if tok.get("outcome", "").lower() == "yes":
                    yes_token_id = tok.get("token_id")
                    break

            # Method 2: parse clobTokenIds (first element = Yes)
            if yes_token_id is None:
                clob_ids = _safe_json_loads(
                    mkt.get("clobTokenIds", "[]"), "clobTokenIds"
                )
                if clob_ids and len(clob_ids) >= 1:
                    yes_token_id = clob_ids[0]

            if yes_token_id:
                bid, ask = self.fetch_order_book(yes_token_id)
                if bid > 0 and ask > 0:
                    prices[team] = (bid + ask) / 2.0
                    continue

            # ---- fallback to outcomePrices ----
            outcome_prices = _safe_json_loads(
                mkt.get("outcomePrices", "[]"), "outcomePrices"
            )
            if outcome_prices and len(outcome_prices) >= 1:
                try:
                    prices[team] = float(outcome_prices[0])
                except (ValueError, IndexError):
                    pass

        logger.info(
            "Fetched midpoint prices for %d teams",
            len(prices),
        )
        return prices

    # ---- public: normalisation ------------------------------------------- #

    @staticmethod
    def normalize_prices(raw_prices: dict[str, float]) -> dict[str, float]:
        """
        Multiplicative normalisation: divide each price by the sum of all
        prices so they sum to 1.0.  This removes the bookmaker overround.
        """
        total = sum(raw_prices.values())
        if total <= 0:
            logger.warning("Total of raw prices is %.4f; cannot normalise", total)
            return raw_prices
        return {team: price / total for team, price in raw_prices.items()}

    # ---- public: main entry point ---------------------------------------- #

    def fetch_market_data(self) -> dict[str, float]:
        """
        Main entry point.

        1. Fetch raw Yes prices from the Gamma API (fast for MVP).
        2. Log the total overround before normalisation.
        3. Normalise to implied probabilities.
        4. Return ``{team_name: implied_probability}``.
        """
        raw_prices = self.fetch_raw_prices()
        if not raw_prices:
            logger.error("No prices fetched — returning empty dict")
            return {}

        overround = sum(raw_prices.values())
        logger.info(
            "Overround before normalisation: %.4f  (%.2f%%)",
            overround,
            (overround - 1.0) * 100,
        )

        normalised = self.normalize_prices(raw_prices)

        # Sanity log
        top5 = sorted(normalised.items(), key=lambda kv: kv[1], reverse=True)[:5]
        logger.info("Top-5 implied probs: %s", top5)

        return normalised

    # ---- cleanup --------------------------------------------------------- #

    def close(self) -> None:
        """Close the underlying httpx client."""
        self._client.close()

    def __enter__(self) -> "MarketFetcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
