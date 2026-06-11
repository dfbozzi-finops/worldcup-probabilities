import math
import numpy as np
from scipy.stats import nbinom, poisson

def calculate_team_props(historical_avg: float, line: float = 4.5, dispersion_factor: float = 1.5) -> dict:
    """
    Calculate Over/Under probabilities for discrete events (Corners, Cards) 
    using the Negative Binomial distribution to account for overdispersion.
    """
    if historical_avg <= 0:
        return {"over": 0.0, "under": 1.0}
        
    variance = historical_avg * dispersion_factor
    
    # nbinom parameters: n (number of successes), p (probability of success)
    # mean = n * (1-p) / p
    # var = n * (1-p) / p^2
    # p = mean / var
    # n = mean^2 / (var - mean)
    
    p = historical_avg / variance
    n_param = (historical_avg ** 2) / (variance - historical_avg)
    
    # Probability of exactly k events
    # We want P(X > line). Using CDF for P(X <= floor(line))
    k = math.floor(line)
    prob_under = nbinom.cdf(k, n_param, p)
    prob_over = 1.0 - prob_under
    
    return {
        "over": round(prob_over, 4),
        "under": round(prob_under, 4)
    }

def calculate_anytime_goalscorer(
    team_xg: float, 
    player_open_play_share: float, 
    is_penalty_taker: bool,
    team_pk_xg: float = 0.15 # Approx penalty xG per match for a top team
) -> float:
    """
    Fractional allocation model.
    Player xG = (Open Play Team xG * Share) + (Penalty Team xG * Penalty Share)
    Returns P(Goals >= 1)
    """
    team_open_play_xg = max(0.0, team_xg - team_pk_xg)
    
    player_xg = (team_open_play_xg * player_open_play_share)
    if is_penalty_taker:
        player_xg += team_pk_xg
        
    # Probability of scoring 0 goals
    p_zero = math.exp(-player_xg)
    
    # P(Goals >= 1)
    p_anytime = 1.0 - p_zero
    return round(p_anytime, 4)
