import httpx
import json
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class APIFootballClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("API_FOOTBALL_KEY")
        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY missing from environment")
            
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            "x-apisports-key": self.api_key,
            "Accept": "application/json"
        }
        self.cache_dir = Path("data/raw/api_football_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _cached_get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request with local file caching."""
        params = params or {}
        # Create a deterministic cache key
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        safe_endpoint = endpoint.replace("/", "_").strip("_")
        cache_filename = f"{safe_endpoint}_{param_str}.json" if param_str else f"{safe_endpoint}.json"
        cache_path = self.cache_dir / cache_filename
        
        if cache_path.exists():
            logger.debug(f"Cache hit for {endpoint} {params}")
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
                
        logger.info(f"API Request: {endpoint} {params}")
        url = f"{self.base_url}{endpoint}"
        response = httpx.get(url, headers=self.headers, params=params, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        # API-Football returns 200 OK even for errors, need to check data['errors']
        if data.get('errors') and not isinstance(data['errors'], list):
            logger.warning(f"API Error Response: {data['errors']}")
            
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        return data
        
    def ping(self) -> dict:
        """Ping the status endpoint to verify authentication and check rate limits."""
        return self._cached_get("/status")
        
    def get_world_cup_leagues(self) -> dict:
        """Search for World Cup league IDs."""
        return self._cached_get("/leagues", {"search": "World Cup"})

    def get_team_search(self, search_name: str) -> dict:
        """Search for a team ID by name."""
        return self._cached_get("/teams", {"search": search_name})

    def get_team_fixtures(self, team_id: int, season: int = 2022) -> dict:
        """Get fixtures for a team in a specific season (Free tier compatible)."""
        return self._cached_get("/fixtures", {"team": team_id, "season": season})

    def get_fixture_statistics(self, fixture_id: int) -> dict:
        """Get team-level event counts (Corners, Cards, etc.) for a specific fixture."""
        return self._cached_get("/fixtures/statistics", {"fixture": fixture_id})

    def get_player_statistics(self, team_id: int, season: int = 2022) -> dict:
        """Get player statistics for a team in a specific season."""
        return self._cached_get("/players", {"team": team_id, "season": season})

def run_diagnostics():
    """Run a diagnostic ping and save the raw JSON to a log file."""
    client = APIFootballClient()
    logger.info("Pinging API-Football /status endpoint...")
    
    diagnostic_data = {}
    
    try:
        status_data = client.ping()
        logger.info("Successfully fetched status data.")
        diagnostic_data['status'] = status_data
        
        logger.info("Fetching World Cup league info...")
        wc_data = client.get_world_cup_leagues()
        logger.info("Successfully fetched World Cup league info.")
        diagnostic_data['world_cup_search'] = wc_data
        
        logger.info("Testing new endpoints: Team Search (Argentina)...")
        team_data = client.get_team_search("Argentina")
        diagnostic_data['team_search'] = team_data
        
        # Try to extract the team ID and get a fixture
        if team_data.get('response'):
            team_id = team_data['response'][0]['team']['id']
            logger.info(f"Found Argentina Team ID: {team_id}. Fetching 2022 season fixtures...")
            fixtures_data = client.get_team_fixtures(team_id, season=2022)
            diagnostic_data['fixtures'] = fixtures_data
            
            # Find a completed fixture
            completed_fixture_id = None
            if fixtures_data.get('response'):
                for f in fixtures_data['response']:
                    if f['fixture']['status']['short'] == 'FT':
                        completed_fixture_id = f['fixture']['id']
                        break
            
            if completed_fixture_id:
                logger.info(f"Found Completed Fixture ID: {completed_fixture_id}. Fetching statistics...")
                stats_data = client.get_fixture_statistics(completed_fixture_id)
                diagnostic_data['fixture_statistics'] = stats_data
                
                logger.info(f"Fetching player statistics for Team {team_id} (Season 2022)...")
                players_data = client.get_player_statistics(team_id, season=2022)
                diagnostic_data['players'] = players_data
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        diagnostic_data['error'] = {
            'status_code': e.response.status_code,
            'message': e.response.text
        }
    except Exception as e:
        logger.error(f"Error: {e}")
        diagnostic_data['error'] = str(e)

    # Save to data/processed/api_diagnostics.log
    out_dir = Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "api_diagnostics.log"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diagnostic_data, f, indent=2)
        
    logger.info(f"Diagnostic payload saved to {out_file}")

if __name__ == "__main__":
    run_diagnostics()
