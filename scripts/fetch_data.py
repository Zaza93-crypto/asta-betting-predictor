import json
import os
import urllib.error
import urllib.parse
import urllib.request
import datetime
from pathlib import Path

TOKEN = os.environ.get("FOOTBALL_DATA_API_TOKEN")
OUT = Path("data/fixtures.json")

if not TOKEN:
    raise SystemExit("ERROR: Missing FOOTBALL_DATA_API_TOKEN GitHub secret")

# These five top European competitions are included in football-data.org's
# Free tier. One combined request keeps us well below the 10 calls/minute limit.
COMPETITIONS = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
}

DAYS_AHEAD = 7

today = datetime.date.today()
date_from = today.isoformat()
date_to = (today + datetime.timedelta(days=DAYS_AHEAD)).isoformat()

params = {
    "competitions": ",".join(COMPETITIONS.keys()),
    "dateFrom": date_from,
    "dateTo": date_to,
    "status": "SCHEDULED",
    "limit": "100",
}

url = "https://api.football-data.org/v4/matches?" + urllib.parse.urlencode(params)

headers = {
    "X-Auth-Token": TOKEN,
    "Accept": "application/json",
}

print("football-data.org live fixture test")
print(f"Date range: {date_from} to {date_to}")
print(f"Competitions: {', '.join(COMPETITIONS.keys())}")
print("-" * 60)

request = urllib.request.Request(url, headers=headers)

try:
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
        http_status = response.status
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    print(f"HTTP ERROR {exc.code}: {body}")
    raise SystemExit(f"football-data.org request failed with HTTP {exc.code}")
except Exception as exc:
    raise SystemExit(f"REQUEST ERROR: {exc}")

if http_status != 200:
    raise SystemExit(f"Unexpected HTTP status: {http_status}")

matches = []

for item in payload.get("matches", []):
    competition = item.get("competition") or {}
    code = competition.get("code")
    league_name = COMPETITIONS.get(code, competition.get("name", "Unknown"))

    home_team = item.get("homeTeam") or {}
    away_team = item.get("awayTeam") or {}

    fixture_id = item.get("id")
    home = home_team.get("name")
    away = away_team.get("name")

    if not fixture_id or not home or not away:
        continue

    matches.append({
        "id": fixture_id,
        "league": league_name,
        "competition_code": code,
        "date": item.get("utcDate"),
        "status": item.get("status"),
        "home": home,
        "away": away,

        # Prediction fields are deliberately left unfilled until
        # Asta's own prediction model is validated.
        "pick": "Model pending",
        "confidence": None,
    })

matches.sort(key=lambda match: match.get("date") or "")

league_report = {}

for code, league_name in COMPETITIONS.items():
    count = sum(1 for match in matches if match["competition_code"] == code)
    league_report[league_name] = {
        "competition_code": code,
        "upcoming_fixtures": count,
    }
    print(f"{league_name}: {count} upcoming fixtures")

data = {
    "source": "football-data.org",
    "updated": datetime.datetime.now(datetime.timezone.utc)
    .replace(microsecond=0)
    .isoformat()
    .replace("+00:00", "Z"),
    "date_range": {
        "from": date_from,
        "to": date_to,
    },
    "league_report": league_report,
    "matches": matches,
    "performance": {
        "tracked": 0,
        "correct": 0,
        "accuracy": None,
    },
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

print("-" * 60)
print(f"API returned {len(payload.get('matches', []))} scheduled matches")
print(f"Wrote {len(matches)} upcoming fixtures to {OUT}")

if matches:
    print("SUCCESS: Current fixture data was retrieved.")
else:
    print("WARNING: No scheduled fixtures were returned for this date range.")

print("Prediction status: Model pending — no unvalidated accuracy claim is being made.")
