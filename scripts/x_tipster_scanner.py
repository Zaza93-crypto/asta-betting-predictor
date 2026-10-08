#!/usr/bin/env python3
"""Asta V3 Tipster Intelligence - read-only X scanner.

Uses the official X API recent-search endpoint to capture explicit football
betting tips. It does NOT post, reply, like, follow, DM, or scrape X.

Required GitHub Actions secret:
  X_BEARER_TOKEN
"""

import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures.json"
TIPS_OUTPUT = ROOT / "data" / "tipster_predictions.json"
TIPSTERS_OUTPUT = ROOT / "data" / "tipsters.json"

API_URL = "https://api.x.com/2/tweets/search/recent"
MAX_RESULTS = 100

MARKET_PATTERNS = [
    (re.compile(r"\b(home win|home\s*w|home)\b", re.I), "HOME WIN"),
    (re.compile(r"\b(away win|away\s*w|away)\b", re.I), "AWAY WIN"),
    (re.compile(r"\b(draw|tie)\b", re.I), "DRAW"),
    (re.compile(r"\b(over|o)\s*2(?:\.5)?\b", re.I), "OVER 2.5"),
    (re.compile(r"\b(under|u)\s*2(?:\.5)?\b", re.I), "UNDER 2.5"),
    (re.compile(r"\b(btts|both teams to score)\b", re.I), "BTTS"),
]

SEARCH_QUERIES = [
    '(football OR soccer) (tip OR tips OR prediction OR picks) (odds OR "to win" OR "over 2.5") -is:retweet -is:reply lang:en',
    '("bet of the day" OR "football tips" OR "soccer tips") (odds OR prediction) -is:retweet -is:reply lang:en',
]

def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc

def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

def normalize(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def api_get(params):
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        raise RuntimeError("X_BEARER_TOKEN is not configured.")
    req = Request(
        f"{API_URL}?{urlencode(params)}",
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "Asta-V3-Tipster-Intelligence/1.0",
        },
    )
    try:
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"X API HTTP {exc.code}: {body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"X API connection failed: {exc}") from exc

def extract_odds(text):
    # Only accept explicitly written decimal odds such as @1.80 or odds 1.80.
    for match in re.finditer(r"(?:@\s*|odds?\s*[:=]?\s*)(\d+(?:\.\d{1,3})?)", text, re.I):
        value = float(match.group(1))
        if 1.01 <= value <= 100:
            return value
    return None

def extract_market(text):
    for pattern, label in MARKET_PATTERNS:
        if pattern.search(text):
            return label
    return None

def fixture_candidates(fixtures):
    rows = []
    for row in fixtures.get("matches", []):
        home = row.get("home") or row.get("home_team")
        away = row.get("away") or row.get("away_team")
        if home and away:
            rows.append({
                "fixture_id": str(row.get("id")),
                "date": row.get("date"),
                "competition": row.get("league") or row.get("competition"),
                "home": home,
                "away": away,
                "home_n": normalize(home),
                "away_n": normalize(away),
            })
    return rows

def find_fixture(text, fixtures):
    ntext = normalize(text)
    best = None
    best_score = 0
    for fixture in fixtures:
        home_words = [w for w in fixture["home_n"].split() if len(w) >= 4]
        away_words = [w for w in fixture["away_n"].split() if len(w) >= 4]
        home_hits = sum(w in ntext for w in home_words)
        away_hits = sum(w in ntext for w in away_words)
        if home_hits and away_hits:
            score = home_hits + away_hits
            if score > best_score:
                best = fixture
                best_score = score
    return best

def extract_tip(post, fixtures):
    text = post.get("text", "")
    market = extract_market(text)
    fixture = find_fixture(text, fixtures)
    if not market or not fixture:
        return None

    return {
        "tip_id": f"x-{post.get('id')}",
        "x_post_id": str(post.get("id")),
        "x_url": f"https://x.com/i/web/status/{post.get('id')}",
        "tipster_username": None,
        "tipster_name": None,
        "posted_at": post.get("created_at"),
        "fixture_id": fixture["fixture_id"],
        "competition": fixture["competition"],
        "home_team": fixture["home"],
        "away_team": fixture["away"],
        "market": market,
        "selection": market,
        "odds_decimal": extract_odds(text),
        "source_text": text,
        "verification_status": "PENDING",
        "result": None,
        "profit_units": None,
        "verified_at": None,
    }

def fetch_posts():
    posts, users = [], {}
    for query in SEARCH_QUERIES:
        data = api_get({
            "query": query,
            "max_results": MAX_RESULTS,
            "tweet.fields": "created_at,author_id,lang,public_metrics",
            "expansions": "author_id",
            "user.fields": "username,name",
        })
        posts.extend(data.get("data", []))
        for user in data.get("includes", {}).get("users", []):
            users[str(user["id"])] = user
    return posts, users

def build_tipsters(tips):
    grouped = {}
    for tip in tips:
        key = tip.get("tipster_username") or "unknown"
        row = grouped.setdefault(key, {
            "username": key,
            "name": tip.get("tipster_name"),
            "tips_tracked": 0,
            "verified_tips": 0,
            "wins": 0,
            "losses": 0,
            "accuracy": None,
            "roi_percent": None,
            "rating": None,
            "classification": "WATCH",
        })
        row["tips_tracked"] += 1
        if tip.get("verification_status") == "VERIFIED":
            row["verified_tips"] += 1
            if tip.get("result") == "WIN":
                row["wins"] += 1
            elif tip.get("result") == "LOSS":
                row["losses"] += 1

    for row in grouped.values():
        if row["verified_tips"]:
            row["accuracy"] = round(row["wins"] / row["verified_tips"] * 100, 2)
        if row["verified_tips"] >= 50:
            row["classification"] = "TRUSTED" if (row["accuracy"] or 0) >= 60 else "WATCH"

    return sorted(
        grouped.values(),
        key=lambda x: (x["accuracy"] is not None, x["accuracy"] or 0),
        reverse=True,
    )

def main():
    if not os.environ.get("X_BEARER_TOKEN", "").strip():
        print("X_BEARER_TOKEN is not configured; scanner skipped.")
        return 0

    fixtures = fixture_candidates(load_json(FIXTURES, {}))
    existing = load_json(TIPS_OUTPUT, {"tips": []})
    tips = existing.get("tips", []) if isinstance(existing, dict) else []
    existing_ids = {row.get("tip_id") for row in tips}

    posts, users = fetch_posts()
    new_tips = []

    for post in posts:
        tip = extract_tip(post, fixtures)
        if not tip or tip["tip_id"] in existing_ids:
            continue
        author = users.get(str(post.get("author_id")), {})
        tip["tipster_username"] = author.get("username")
        tip["tipster_name"] = author.get("name")
        new_tips.append(tip)

    tips.extend(new_tips)
    now = datetime.now(timezone.utc).isoformat()

    save_json(TIPS_OUTPUT, {
        "updated_at": now,
        "model_version": "asta-v3-final-walkforward-logistic",
        "source": "X API recent search",
        "read_only": True,
        "tips": tips[-5000:],
    })

    save_json(TIPSTERS_OUTPUT, {
        "updated_at": now,
        "source": "Derived from tipster_predictions.json",
        "tipsters": build_tipsters(tips),
    })

    print(f"X posts scanned: {len(posts)}")
    print(f"New explicit tips captured: {len(new_tips)}")
    print(f"Total stored tips: {len(tips)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
