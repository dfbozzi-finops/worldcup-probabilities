import json
import logging
from itertools import combinations
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_match_by_match_json(dc_model, groups_dict: dict[str, list[str]], output_path: str = "data/processed/match_by_match.json"):
    """
    Generate match-by-match probabilities (1X2, Over/Under 1.5, 2.5, BTTS) 
    for all group stage matches and export to JSON.
    """
    matches = []
    
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
                        
            match_data = {
                "group": group_name,
                "home_team": home,
                "away_team": away,
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
            matches.append(match_data)
            
    # Export to JSON
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(matches, f, indent=2)
        
    logger.info(f"Generated match-by-match probabilities for {len(matches)} matches at {output_path}")
    return matches
