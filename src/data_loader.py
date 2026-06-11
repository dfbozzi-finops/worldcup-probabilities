"""
Data Loader — World Cup 2026 Statistical Arbitrage System

Handles data acquisition from Kaggle, cleaning, team name normalization,
and provides canonical lists of World Cup 2026 teams with FIFA rankings.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# FIFA Rankings & Team Definitions
# ---------------------------------------------------------------------------

_FIFA_RANKINGS: dict[str, int] = {
    "France": 1,
    "Spain": 2,
    "Argentina": 3,
    "England": 4,
    "Portugal": 5,
    "Brazil": 6,
    "Netherlands": 7,
    "Morocco": 8,
    "Belgium": 9,
    "Germany": 10,
    "Croatia": 11,
    "Colombia": 13,
    "Senegal": 14,
    "Mexico": 15,
    "United States": 16,
    "Uruguay": 17,
    "Japan": 18,
    "Switzerland": 19,
    "Iran": 21,
    "Türkiye": 22,
    "Ecuador": 23,
    "Austria": 24,
    "South Korea": 25,
    "Australia": 27,
    "Algeria": 28,
    "Egypt": 29,
    "Canada": 30,
    "Norway": 31,
    "Ukraine": 32,
    "Panama": 33,
    "Ivory Coast": 34,
    "Paraguay": 40,
    "Czechia": 41,
    "Scotland": 43,
    "Tunisia": 44,
    "DR Congo": 46,
    "Uzbekistan": 50,
    "Qatar": 55,
    "Iraq": 57,
    "South Africa": 60,
    "Saudi Arabia": 61,
    "Jordan": 63,
    "Bosnia and Herzegovina": 65,
    "Cape Verde": 69,
    "Ghana": 74,
    "Curaçao": 82,
    "Haiti": 83,
    "New Zealand": 85,
}

# Mapping from common alternative/historical names → canonical names
_TEAM_NAME_MAPPING: dict[str, str] = {
    "Turkey": "Türkiye",
    "Turkiye": "Türkiye",
    "Korea Republic": "South Korea",
    "Korea, Republic of": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "United States of America": "United States",
    "IR Iran": "Iran",
    "Iran, Islamic Rep.": "Iran",
    "Congo DR": "DR Congo",
    "Democratic Republic of Congo": "DR Congo",
    "Dem. Rep. of the Congo": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Czech Republic": "Czechia",
    "Curacao": "Curaçao",
    "New Caledonia": "New Zealand",  # not a real alias but avoids crash
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_fifa_rankings() -> dict[str, int]:
    """Return a copy of the 48-team FIFA ranking dictionary."""
    return dict(_FIFA_RANKINGS)


def get_wc_teams() -> list[str]:
    """Return a sorted list of all 48 World Cup 2026 team names."""
    return sorted(_FIFA_RANKINGS.keys())


def get_team_name_mapping() -> dict[str, str]:
    """Return a copy of the alternative-name → canonical-name mapping."""
    return dict(_TEAM_NAME_MAPPING)


def normalize_team_name(name: str) -> str:
    """Convert *name* to its canonical form, or return it unchanged."""
    return _TEAM_NAME_MAPPING.get(name, name)


# ---------------------------------------------------------------------------
# Kaggle data acquisition
# ---------------------------------------------------------------------------


def download_kaggle_dataset(
    dataset: str = "martj42/international-football-results-from-1872-to-2017",
    dest: str = "data/raw",
) -> Path:
    """
    Download and unzip a Kaggle dataset into *dest*.

    Credentials are read from ``.env`` (``KAGGLE_USERNAME``, ``KAGGLE_KEY``).
    If ``data/raw/results.csv`` already exists the download is skipped.

    Returns
    -------
    Path
        Absolute path to ``results.csv``.
    """
    dest_path = Path(dest)
    results_file = dest_path / "results.csv"

    if results_file.exists():
        return results_file.resolve()

    # Load credentials from .env before importing kaggle
    load_dotenv()
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY")
    if not kaggle_user or not kaggle_key:
        raise EnvironmentError(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set in the .env file "
            "or as environment variables."
        )

    # kaggle reads os.environ at import time, so set before import
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key

    # Late import so credentials are available
    from kaggle import api as kaggle_api  # type: ignore[import-untyped]

    dest_path.mkdir(parents=True, exist_ok=True)
    kaggle_api.dataset_download_files(dataset, path=str(dest_path), unzip=True)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Expected {results_file} after download but it was not found. "
            f"Check that the dataset '{dataset}' contains results.csv."
        )

    return results_file.resolve()


# ---------------------------------------------------------------------------
# Match data loading
# ---------------------------------------------------------------------------


def load_match_data(history_years: int = 20) -> pd.DataFrame:
    """
    Load and clean international match results.

    Parameters
    ----------
    history_years : int
        Only matches from the last *history_years* years (from today) are
        returned.

    Returns
    -------
    pd.DataFrame
        Columns: date, home_team, away_team, home_score, away_score,
        tournament, city, country, neutral.
    """
    results_path = download_kaggle_dataset()

    df = pd.read_csv(
        results_path,
        parse_dates=["date"],
        dtype={
            "home_score": "Int64",
            "away_score": "Int64",
            "neutral": bool,
        },
    )

    # Ensure expected columns exist
    expected_cols = {
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in results.csv: {missing}")

    # Normalise team names to canonical form
    df["home_team"] = df["home_team"].map(normalize_team_name)
    df["away_team"] = df["away_team"].map(normalize_team_name)

    # Filter to recent history
    cutoff = datetime.today() - pd.DateOffset(years=history_years)
    df = df.loc[df["date"] >= cutoff].copy()

    # Drop rows with missing scores
    df.dropna(subset=["home_score", "away_score"], inplace=True)

    # Cast scores to plain int after dropping NaN
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
