import json
import os
import urllib.error
import urllib.request
import datetime
from pathlib import Path

KEY = os.environ.get("API_FOOTBALL_KEY")
OUT = Path("data/fixtures.json")

if not KEY:
    raise SystemExit("ERROR: Missing API_FOOTBALL_KEY GitHub secret")

# Major leagues for the first live-data test.
LEAGUES = {
    39: "Premier League",
    140: "La Liga",
    135: "Serie A",
    78: "Bundesliga",
    61: "Ligue 1",
}

SEASON = 2026
DAYS_AHEAD = 7
MAX_PREDICTIONS = 20

HEADERS = {
    "x-apisports-key": KEY,
    "Accept": "application/json",
}

today = datetime.date.today()
date_from = today.isoformat()
date_to = (today + datetime.timedelta(days=DAYS_AHEAD)).isoformat()


def get_json(url):
    """Call API-Football and return the decoded JSON plus HTTP status."""
    request = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body), response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP ERROR {exc.code}: {body}")
        return None, exc.code
    except Exception as exc:
        print(f"REQUEST ERROR: {exc}")
        return None, None


matches = []
league_report = []
prediction_requests = 0

print(f"API-Football live data test")
print(f"Date range: {date_from} to {date_to}")
print(f"Season: {SEASON}")
print("-" * 60)

for league_id, league_name in LEAGUES.items():
    url = (
        "https://v3.football.api-sports.io/fixtures"
        f"?league={league_id}&season={SEASON}"
        f"&from={date_from}&to={date_to}"
    )

    payload, status = get_json(url)

    if payload is None:
        league_report.append({
            "league": league_name,
            "status": status,
            "fixtures": 0,
            "error": "Request failed; see workflow log",
        })
        continue

    errors = payload.get("errors") or {}
    response = payload.get("response") or []

    if errors:
        print(f"{league_name}: API errors -> {errors}")
        league_report.append({
            "league": league_name,
            "status": status,
            "fixtures": 0,
            "error": errors,
        })
        continue

    print(f"{league_name}: API returned {len(response)} fixture records")

    added = 0

    for item in response:
        fixture = item.get("fixture", {})
        teams = item.get("teams", {})
        fixture_status = (fixture.get("status") or {}).get("short")

        # Only upcoming fixtures.
        if fixture_status not in ("NS", "TBD"):
            continue

        fixture_id = fixture.get("id")
        home = (teams.get("home") or {}).get("name")
        away = (teams.get("away") or {}).get("name")

        if not fixture_id or not home or not away:
            continue

        pick = "No pick"
        confidence = 0
        prediction_error = None

        # Limit prediction calls so the free API quota is not unnecessarily consumed.
        if prediction_requests < MAX_PREDICTIONS:
            prediction_url = (
                "https://v3.football.api-sports.io/predictions"
                f"?fixture={fixture_id}"
            )
            prediction_payload, prediction_status = get_json(prediction_url)
            prediction_requests += 1

            if prediction_payload is None:
                prediction_error = f"HTTP/request failure: {prediction_status}"
            else:
                prediction_errors = prediction_payload.get("errors") or {}

                if prediction_errors:
                    prediction_error = prediction_errors
                else:
                    prediction_response = prediction_payload.get("response") or []

                    if prediction_response:
                        prediction = prediction_response[0].get("predictions") or {}
                        winner = prediction.get("winner") or {}
                        winner_name = winner.get("name")

                        if winner_name:
                            pick = winner_name

                        percent = prediction.get("percent") or {}
                        values = []

                        for value in percent.values():
                            try:
                                values.append(
                                    float(str(value).replace("%", "").strip())
                                )
                            except (TypeError, ValueError):
                                pass

                        if values:
                            confidence = round(max(values))

        matches.append({
            "id": fixture_id,
            "league": league_name,
            "date": fixture.get("date"),
            "home": home,
            "away": away,
            "pick": pick,
            "confidence": confidence,
            "prediction_error": prediction_error,
        })

        added += 1

    league_report.append({
        "league": league_name,
        "status": status,
        "fixtures": len(response),
        "upcoming_added": added,
        "error": None,
    })

    print(f"{league_name}: added {added} upcoming fixtures")

matches.sort(key=lambda match: match.get("date") or "")

data = {
    "source": "api-football",
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
print(f"Prediction requests used: {prediction_requests}")
print(f"Wrote {len(matches)} upcoming fixtures to {OUT}")

if not matches:
    print("WARNING: No upcoming fixtures were found.")
    print("Review the league_report above for API errors or empty league results.")
else:
    print("SUCCESS: Live fixture data was retrieved.")
