#!/usr/bin/env python3
"""
Asta Football Predictor v2.

Feature-based baseline using information available before a match:
- sequential Elo/team strength
- recent points per match
- recent goals for/against
- home/away-specific recent performance
- recency weighting

For live fixtures, data/historical_matches.json is used as the historical
knowledge base. If it is unavailable, the model falls back to neutral features.

This is a predictive baseline, not a guarantee of accuracy.
"""

import json
import math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "data" / "fixtures.json"
HISTORY = ROOT / "data" / "historical_matches.json"
RATINGS = ROOT / "data" / "team_ratings.json"
OUTPUT = ROOT / "data" / "predictions.json"


def load(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def norm(s):
    return " ".join(str(s or "").strip().lower().split())


def team_name(match, side):
    if side == "home":
        candidates = ["home_team", "homeTeam", "home", "home_name"]
        nested = ["homeTeam", "home"]
    else:
        candidates = ["away_team", "awayTeam", "away", "away_name"]
        nested = ["awayTeam", "away"]
    for k in candidates:
        if match.get(k):
            return str(match[k])
    for k in nested:
        if isinstance(match.get(k), dict):
            return str(match[k].get("name") or match[k].get("team") or "")
    return ""


def competition(match):
    c = match.get("competition") or match.get("league") or ""
    return str(c.get("name") if isinstance(c, dict) else c)


def fixture_id(match, i):
    return str(match.get("id") or match.get("fixture_id") or match.get("match_id") or f"fixture-{i+1}")


def parse_history(payload):
    matches = payload.get("matches", []) if isinstance(payload, dict) else payload
    out = []
    for m in matches:
        s = m.get("score") or {}
        h, a = s.get("home"), s.get("away")
        if not isinstance(h, int) or not isinstance(a, int):
            continue
        home, away = m.get("home_team"), m.get("away_team")
        if not home or not away:
            continue
        out.append({
            "date": m.get("date") or "",
            "home": home, "away": away,
            "hg": h, "ag": a,
            "competition": m.get("competition", "")
        })
    out.sort(key=lambda x: x["date"])
    return out


def build_features(history, ratings):
    # Each team keeps only its latest 10 matches.
    recent = defaultdict(lambda: deque(maxlen=10))
    home_recent = defaultdict(lambda: deque(maxlen=8))
    away_recent = defaultdict(lambda: deque(maxlen=8))
    elo = defaultdict(lambda: 1500.0)

    for m in history:
        h, a, hg, ag = m["home"], m["away"], m["hg"], m["ag"]

        # Update team histories AFTER this match.
        recent[h].append((1 if hg > ag else 2 if hg == ag else 0, hg, ag))
        recent[a].append((1 if ag > hg else 2 if hg == ag else 0, ag, hg))
        home_recent[h].append((1 if hg > ag else 2 if hg == ag else 0, hg, ag))
        away_recent[a].append((1 if ag > hg else 2 if hg == ag else 0, ag, hg))

        expected = 1 / (1 + 10 ** (-((elo[h] + 55) - elo[a]) / 400))
        actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        change = 20 * (actual - expected)
        elo[h] += change
        elo[a] -= change

    # Seed from optional ratings only when they exist and team has no learned Elo.
    for name, value in ratings.items():
        try:
            if norm(name) not in {norm(k) for k in elo.keys()}:
                elo[name] = float(value)
        except Exception:
            pass

    def lookup(store, team):
        key = norm(team)
        for name, values in store.items():
            if norm(name) == key:
                return values
        return []

    return recent, home_recent, away_recent, elo, lookup


def avg(values, index, default):
    if not values:
        return default
    return sum(v[index] for v in values) / len(values)


def feature_vector(team, recent, home_recent, away_recent, elo, lookup, venue):
    r = lookup(recent, team)
    venue_data = lookup(home_recent if venue == "home" else away_recent, team)

    points = avg(r, 0, 1.0)
    gf = avg(r, 1, 1.35)
    ga = avg(r, 2, 1.35)
    vpoints = avg(venue_data, 0, 1.0)
    vgf = avg(venue_data, 1, 1.35)
    vga = avg(venue_data, 2, 1.35)

    rating = 1500.0
    for name, value in elo.items():
        if norm(name) == norm(team):
            rating = float(value)
            break

    return {
        "elo": rating,
        "points": points,
        "gf": gf,
        "ga": ga,
        "venue_points": vpoints,
        "venue_gf": vgf,
        "venue_ga": vga,
    }


def predict(home, away, features):
    hf, af = features
    elo_diff = ((hf["elo"] + 55) - af["elo"]) / 400.0

    # Goal expectation. These coefficients are intentionally modest:
    # v2 is a baseline to validate, not an overfit model.
    attack = 0.45 * (hf["gf"] - af["ga"]) + 0.35 * (hf["venue_gf"] - af["venue_ga"])
    defense = 0.35 * (af["gf"] - hf["ga"]) + 0.25 * (af["venue_gf"] - hf["venue_ga"])
    form = 0.22 * (hf["points"] - af["points"]) + 0.18 * (hf["venue_points"] - af["venue_points"])

    home_score = 0.95 + 0.75 * elo_diff + 0.18 * attack + 0.12 * form
    away_score = 0.95 - 0.75 * elo_diff + 0.18 * defense - 0.12 * form

    home_score = max(0.20, min(3.80, home_score))
    away_score = max(0.20, min(3.80, away_score))

    # Poisson score grid, 0-6 goals.
    probs = {"home_win": 0.0, "draw": 0.0, "away_win": 0.0}
    for hg in range(7):
        for ag in range(7):
            p_h = math.exp(-home_score) * home_score ** hg / math.factorial(hg)
            p_a = math.exp(-away_score) * away_score ** ag / math.factorial(ag)
            p = p_h * p_a
            if hg > ag:
                probs["home_win"] += p
            elif hg == ag:
                probs["draw"] += p
            else:
                probs["away_win"] += p

    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def confidence(probs):
    vals = sorted(probs.values(), reverse=True)
    top, second = vals[0], vals[1]
    if top >= 0.68 and top - second >= 0.20:
        return "HIGH"
    if top >= 0.56 and top - second >= 0.10:
        return "MEDIUM"
    return "LOW"


def main():
    fixtures = load(FIXTURES, [])
    if isinstance(fixtures, dict):
        fixtures = fixtures.get("fixtures") or fixtures.get("matches") or fixtures.get("data") or []

    history = parse_history(load(HISTORY, {"matches": []}))
    ratings = load(RATINGS, {})

    recent, home_recent, away_recent, elo, lookup = build_features(history, ratings)

    labels = {"home_win": "HOME WIN", "draw": "DRAW", "away_win": "AWAY WIN"}
    output = []

    for i, match in enumerate(fixtures):
        home, away = team_name(match, "home"), team_name(match, "away")
        if not home or not away:
            continue

        hf = feature_vector(home, recent, home_recent, away_recent, elo, lookup, "home")
        af = feature_vector(away, recent, home_recent, away_recent, elo, lookup, "away")
        p = predict(home, away, (hf, af))
        key = max(p, key=p.get)

        output.append({
            "fixture_id": fixture_id(match, i),
            "date": match.get("date") or match.get("utcDate") or match.get("match_date"),
            "competition": competition(match),
            "home_team": home,
            "away_team": away,
            "probabilities": {k: round(v * 100, 2) for k, v in p.items()},
            "prediction": labels[key],
            "confidence": confidence(p),
            "model_version": "asta-v2-form-elo-poisson",
            "validated_accuracy": None,
            "status": "PENDING_RESULT"
        })

    result = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": "asta-v2-form-elo-poisson",
        "accuracy_status": "NOT_VALIDATED",
        "accuracy_note": "Use the chronological backtest before making accuracy claims.",
        "historical_features_available": bool(history),
        "total_predictions": len(output),
        "predictions": output
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Generated {len(output)} v2 predictions.")
    print("Model status: NOT VALIDATED")


if __name__ == "__main__":
    main()
