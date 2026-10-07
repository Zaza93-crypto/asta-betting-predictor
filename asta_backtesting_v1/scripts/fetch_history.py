#!/usr/bin/env python3
"""
Fetch historical football results from football-data.org.

The API token is read from FOOTBALL_DATA_API_TOKEN.
Historical data is saved to data/historical_matches.json.

Default period: 2025-08-01 through 2026-05-31.
This is intentionally separate from the live fixture pipeline.
"""

import json
import os
import time
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "historical_matches.json"

COMPETITIONS = ["PL", "PD", "SA", "BL1", "FL1"]
DATE_FROM = "2025-08-01"
DATE_TO = "2026-05-31"


def get_json(url, token):
    req = Request(url, headers={
        "X-Auth-Token": token,
        "User-Agent": "Asta-Football-Predictor/1.0",
    })
    with urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    token = os.environ.get("FOOTBALL_DATA_API_TOKEN")
    if not token:
        raise RuntimeError("FOOTBALL_DATA_API_TOKEN secret is missing.")

    all_matches = []
    errors = []

    for code in COMPETITIONS:
        url = (
            f"https://api.football-data.org/v4/competitions/{code}/matches"
            f"?dateFrom={DATE_FROM}&dateTo={DATE_TO}"
        )
        try:
            payload = get_json(url, token)
            matches = payload.get("matches", [])
            for match in matches:
                all_matches.append({
                    "competition": code,
                    "date": match.get("utcDate"),
                    "home_team": (match.get("homeTeam") or {}).get("name"),
                    "away_team": (match.get("awayTeam") or {}).get("name"),
                    "status": match.get("status"),
                    "score": match.get("score", {}).get("fullTime", {}),
                    "fixture_id": str(match.get("id")),
                })
            print(f"{code}: {len(matches)} historical matches")
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            errors.append(f"{code}: {exc}")
            print(f"WARNING: {code} failed: {exc}")
        time.sleep(1)

    all_matches.sort(key=lambda x: x.get("date") or "")

    output = {
        "generated_at": date.today().isoformat(),
        "date_from": DATE_FROM,
        "date_to": DATE_TO,
        "competitions": COMPETITIONS,
        "total_matches": len(all_matches),
        "errors": errors,
        "matches": all_matches,
    }

    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Wrote {len(all_matches)} historical matches to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
