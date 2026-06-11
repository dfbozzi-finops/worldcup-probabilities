import json
import logging
from pathlib import Path
from src.api_football import APIFootballClient
from src.fbref_scraper import get_fbref_world_cup_stats, get_fbref_player_stat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def fetch_props_data_sample():
    """
    Fetch and cache team statistics (Cards, Corners) and player metrics 
    for a sample of active World Cup teams to verify the data pipeline 
    without busting the 100 req/day limit.
    """
    client = APIFootballClient()
    
    # We will test with Argentina (Team ID 26) and France (Team ID 67) for the 2022 World Cup
    test_teams = {
        "Argentina": 26,
        "France": 67
    }
    
    pipeline_data = {}
    
    for team_name, team_id in test_teams.items():
        logger.info(f"Processing {team_name} (ID: {team_id})...")
        team_data = {}
        
        # 1. Fetch Fixtures
        fixtures = client.get_team_fixtures(team_id, season=2022)
        if not fixtures.get('response'):
            logger.warning(f"No fixtures found for {team_name}")
            continue
            
        # Get just the first completed fixture to extract team-level event counts
        completed_fixtures = [f for f in fixtures['response'] if f['fixture']['status']['short'] == 'FT']
        if completed_fixtures:
            fixture_id = completed_fixtures[0]['fixture']['id']
            logger.info(f"  Fetching statistics for Fixture {fixture_id}...")
            stats = client.get_fixture_statistics(fixture_id)
            
            # Extract Cards and Corners
            if stats.get('response'):
                for team_stat in stats['response']:
                    if team_stat['team']['id'] == team_id:
                        extracted = {}
                        for stat in team_stat['statistics']:
                            if stat['type'] in ['Corner Kicks', 'Yellow Cards', 'Red Cards', 'Fouls']:
                                # Coerce null to 0 ONLY for match events
                                val = stat['value']
                                extracted[stat['type']] = 0 if val is None else val
                        team_data['sample_fixture_stats'] = extracted
                        logger.info(f"  Extracted Stats: {extracted}")
        
        # 2. Fetch Player Statistics for offensive metrics
        logger.info(f"  Fetching player statistics for {team_name}...")
        players = client.get_player_statistics(team_id, season=2022)
        
        # Load FBref fallback dataframe
        fbref_df = get_fbref_world_cup_stats()
        
        if players.get('response'):
            extracted_players = []
            
            # Since Messi might not be in the first 2 results due to pagination,
            # Let's search all returned players, or specifically find him.
            # API-Football paginates players. For this test, we just look at the returned page.
            for p in players['response']:
                player_name = p['player']['name']
                # Only keep players we care about for the diagnostic (Messi, Mbappe, etc.)
                # or just process everyone in the page.
                if 'Messi' in player_name or 'Correa' in player_name or 'Mbapp' in player_name:
                    stats = p['statistics'][0] # World Cup stats
                    
                    # Extract raw API stats
                    goals = stats['goals']['total']
                    assists = stats['goals']['assists']
                    shots_on = stats['shots']['on']
                    pk_scored = stats['penalty']['scored']
                    pk_missed = stats['penalty']['missed']
                    
                    # Apply FBref fallback as the primary source for historical aggregates 
                    # due to API-Football DB inaccuracies (e.g. returning 2 instead of 4 for penalties)
                    fbref_assists = get_fbref_player_stat(fbref_df, player_name, 'Ast')
                    if fbref_assists is not None:
                        assists = fbref_assists
                    elif assists is None:
                        logger.info(f"  Missing assists for {player_name} and not in FBref. Leaving as None.")
                    
                    fbref_goals = get_fbref_player_stat(fbref_df, player_name, 'Gls')
                    if fbref_goals is not None:
                        goals = fbref_goals
                        
                    fbref_pk_scored = get_fbref_player_stat(fbref_df, player_name, 'PK')
                    if fbref_pk_scored is not None:
                        pk_scored = fbref_pk_scored

                    offensive_stats = {
                        "goals": goals,
                        "assists": assists,
                        "shots_on": shots_on,
                        "penalty_scored": pk_scored,
                        "penalty_missed": pk_missed
                    }
                    extracted_players.append({player_name: offensive_stats})
                    
            team_data['sample_player_stats'] = extracted_players
            logger.info(f"  Extracted Players: {extracted_players}")
            
        pipeline_data[team_name] = team_data
        
    # Save diagnostic output
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "props_pipeline_diagnostics.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pipeline_data, f, indent=2)
        
    logger.info(f"\nProps pipeline verification data saved to {out_file}")

if __name__ == "__main__":
    fetch_props_data_sample()
