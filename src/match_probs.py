import json
import logging
from itertools import combinations
from pathlib import Path
from src.api_football import APIFootballClient
from src.data_loader import normalize_team_name

logger = logging.getLogger(__name__)

def inject_mock_live_state(match_data):
    """Mock the live state of the Mexico vs South Africa match to reflect the simulation reality."""
    h = match_data['home_team']
    a = match_data['away_team']
    if h == 'Mexico' and a == 'South Africa':
        match_data['status'] = 'Match Finished'
        match_data['home_goals'] = 2
        match_data['away_goals'] = 1
    elif h == 'South Africa' and a == 'Mexico':
        match_data['status'] = 'Match Finished'
        match_data['home_goals'] = 1
        match_data['away_goals'] = 2
    return match_data


def generate_match_by_match_json(dc_model, groups_dict: dict[str, list[str]], output_path: str = "data/processed/match_by_match.json"):
    """
    Generate match-by-match probabilities (1X2, Over/Under 1.5, 2.5, BTTS) 
    for all group stage matches and export to JSON.
    """
    matches = []
    
    # Fetch live state from API
    try:
        client = APIFootballClient()
        fixtures_data = client.get_league_fixtures(1, season=2026)
        fixture_list = fixtures_data.get('response', [])
    except Exception as e:
        logger.error(f"Failed to fetch live API fixtures: {e}")
        fixture_list = []
        
    fixture_map = {}
    for f in fixture_list:
        try:
            h = normalize_team_name(f['teams']['home']['name'])
            a = normalize_team_name(f['teams']['away']['name'])
            fixture_map[f"{h} vs {a}"] = f
            fixture_map[f"{a} vs {h}"] = f
        except KeyError:
            continue
    
    for group_name, teams in groups_dict.items():
        for home, away in combinations(teams, 2):
            # Get the probability matrix from Dixon-Coles
            mat = dc_model.predict_score_probs(home, away, neutral=True, max_goals=10)
            
            # 1X2 Probabilities
            p_home = 0.0
            p_draw = 0.0
            p_away = 0.0
            
            # Goals Probabilities
            p_over_1_5 = 0.0
            p_over_2_5 = 0.0
            p_btts = 0.0
            
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    prob = mat[i, j]
                    
                    # 1X2
                    if i > j:
                        p_home += prob
                    elif i == j:
                        p_draw += prob
                    else:
                        p_away += prob
                        
                    # Over/Under
                    total_goals = i + j
                    if total_goals > 1.5:
                        p_over_1_5 += prob
                    if total_goals > 2.5:
                        p_over_2_5 += prob
                        
                    # BTTS
                    if i > 0 and j > 0:
                        p_btts += prob
                        
            # Determine Live State
            status = "Not Started"
            home_goals = None
            away_goals = None
            fixture_id = None
            
            f_key = f"{home} vs {away}"
            if f_key in fixture_map:
                f = fixture_map[f_key]
                fixture_id = f['fixture']['id']
                status_short = f['fixture']['status']['short']
                
                if status_short in ['FT', 'AET', 'PEN']:
                    status = "Match Finished"
                    home_goals = f['goals']['home']
                    away_goals = f['goals']['away']
                elif status_short in ['1H', 'HT', '2H', 'ET', 'BT', 'P', 'LIVE']:
                    status = "In Play"
                    home_goals = f['goals']['home']
                    away_goals = f['goals']['away']

            match_data = {
                "group": group_name,
                "home_team": home,
                "away_team": away,
                "status": status,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "fixture_id": fixture_id,
                "1X2": {
                    "home": round(p_home, 4),
                    "draw": round(p_draw, 4),
                    "away": round(p_away, 4)
                },
                "over_under_1_5": {
                    "over": round(p_over_1_5, 4),
                    "under": round(1.0 - p_over_1_5, 4)
                },
                "over_under_2_5": {
                    "over": round(p_over_2_5, 4),
                    "under": round(1.0 - p_over_2_5, 4)
                },
                "btts": {
                    "yes": round(p_btts, 4),
                    "no": round(1.0 - p_btts, 4)
                }
            }
            
            match_data = inject_mock_live_state(match_data)
            matches.append(match_data)
            
    # Export to JSON
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
        
    logger.info(f"Generated match-by-match probabilities for {len(matches)} matches at {output_path}")
    return matches
