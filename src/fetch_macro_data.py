"""
fetch_macro_data.py — Pulls macroeconomic factors (GDP, Population) 
from the World Bank API to augment the Statistical Arbitrage System's ML models.
Aggressively caches to disk to avoid rate limits.
"""

import json
import logging
import os
from pathlib import Path

import httpx
import pandas as pd
from src.data_loader import normalize_team_name, get_wc_teams

logger = logging.getLogger(__name__)

# World Bank API Endpoints
WB_API_BASE = "https://api.worldbank.org/v2/country/all/indicator"
INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "population": "SP.POP.TOTL"
}

# Static climate approximation (Average yearly temp in Celsius) for the 48 teams
CLIMATE_DATA = {
    "Argentina": 14.8, "Brazil": 25.0, "Spain": 13.3, "England": 8.4, 
    "France": 10.7, "Colombia": 24.5, "Portugal": 15.1, "Germany": 8.5, 
    "Belgium": 9.8, "Morocco": 17.1, "Netherlands": 9.2, "Uruguay": 17.5, 
    "Croatia": 10.9, "Ecuador": 21.8, "Switzerland": 5.5, "Mexico": 21.0, 
    "Japan": 11.1, "Senegal": 27.8, "United States": 8.5, "Iran": 17.2, 
    "Austria": 6.3, "Türkiye": 11.1, "Norway": 1.5, "South Korea": 11.5, 
    "Australia": 21.6, "Algeria": 22.8, "Egypt": 22.1, "Canada": -5.3, 
    "Paraguay": 23.5, "Ivory Coast": 26.3, "Czechia": 7.5, "Ukraine": 8.3, 
    "Scotland": 7.0, "Panama": 26.7, "Tunisia": 19.2, "DR Congo": 24.0, 
    "Uzbekistan": 12.0, "Saudi Arabia": 24.6, "South Africa": 17.5, 
    "Jordan": 18.3, "Bosnia and Herzegovina": 9.8, "Cape Verde": 23.3, 
    "Ghana": 27.2, "New Zealand": 10.5, "Curaçao": 27.4, "Haiti": 24.5,
    "Iraq": 21.4, "Qatar": 27.1
}

def _fetch_wb_indicator(indicator_code: str, year: int = 2023) -> dict[str, float]:
    """Fetch World Bank data for a specific indicator."""
    url = f"{WB_API_BASE}/{indicator_code}"
    params = {
        "format": "json",
        "date": str(year),
        "per_page": 300  # enough for all countries
    }
    
    with httpx.Client(follow_redirects=True, timeout=30.0) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        
    data = resp.json()
    if len(data) < 2:
        logger.error("Unexpected World Bank API response structure.")
        return {}
        
    records = data[1]
    result = {}
    for row in records:
        country_raw = row["country"]["value"]
        # Custom tweaks for WB country names
        if country_raw == "Egypt, Arab Rep.": country_raw = "Egypt"
        elif country_raw == "Iran, Islamic Rep.": country_raw = "Iran"
        elif country_raw == "Korea, Rep.": country_raw = "South Korea"
        elif country_raw == "Turkiye": country_raw = "Türkiye"
        elif country_raw == "Congo, Dem. Rep.": country_raw = "DR Congo"
        elif country_raw == "Slovak Republic": country_raw = "Slovakia"
        elif country_raw == "Russian Federation": country_raw = "Russia"
        elif country_raw == "United Kingdom": country_raw = "England" # Proxy for UK parts
        
        val = row.get("value")
        if val is not None:
            norm_team = normalize_team_name(country_raw)
            result[norm_team] = float(val)
            
    return result

def build_macro_dataset(dest_file: str = "data/raw/macro.csv") -> pd.DataFrame:
    """Build and aggressively cache the macro features dataset."""
    dest_path = Path(dest_file)
    if dest_path.exists():
        logger.info(f"Loading cached macro data from {dest_path}")
        return pd.read_csv(dest_path)
        
    logger.info("Fetching macro data from World Bank API...")
    
    gdp_data = _fetch_wb_indicator(INDICATORS["gdp_per_capita"], year=2022)
    pop_data = _fetch_wb_indicator(INDICATORS["population"], year=2022)
    
    # We use 2022 data as 2023 might be incomplete for some countries.
    
    teams = get_wc_teams()
    rows = []
    
    for team in teams:
        gdp = gdp_data.get(team)
        if gdp is None:
            # Fallback heuristic for missing countries (e.g. Scotland -> England proxy)
            gdp = gdp_data.get("England", 45000.0)
            
        pop = pop_data.get(team)
        if pop is None:
            pop = pop_data.get("England", 60000000.0)
            
        climate = CLIMATE_DATA.get(team, 15.0)
        
        rows.append({
            "team": team,
            "gdp_per_capita": gdp,
            "population": pop,
            "avg_temp_celsius": climate
        })
        
    df = pd.DataFrame(rows)
    
    # Ensure directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_path, index=False)
    logger.info(f"Macro data fully built and cached to {dest_path}")
    
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = build_macro_dataset()
    print(df.head())
