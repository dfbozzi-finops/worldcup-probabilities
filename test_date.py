import json
import requests
from datetime import datetime, timezone, timedelta

response = requests.get("https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json")
data = response.json()

group_matches = [m for m in data["matches"] if "Matchday" in m["round"]]
print(f"Total group matches: {len(group_matches)}")

def parse_time(date_str, time_str):
    # e.g. "2026-06-11", "13:00 UTC-6"
    # Or "12:00 UTC-4"
    if "UTC" in time_str:
        time_part, tz_part = time_str.split(" UTC")
        hours = int(tz_part)
        dt = datetime.strptime(f"{date_str} {time_part}", "%Y-%m-%d %H:%M")
        dt = dt.replace(tzinfo=timezone(timedelta(hours=hours)))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return None

for m in group_matches[:5]:
    iso = parse_time(m["date"], m["time"])
    print(m["date"], m["time"], "->", iso)
