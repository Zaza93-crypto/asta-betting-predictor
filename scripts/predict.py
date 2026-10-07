#!/usr/bin/env python3
"""
Asta Football Predictor - Phase 2 prediction engine.

This version is deliberately conservative:
- Reads current fixtures from data/fixtures.json.
- Uses team-strength ratings when available in data/team_ratings.json.
- Falls back to neutral ratings when historical team data is unavailable.
- Produces Home/Draw/Away probabilities that sum to 100%.
- Applies a confidence label without claiming a proven accuracy rate.

To improve the model, add validated team ratings/statistics to data/team_ratings.json
and later connect historical match results for back-testing.
"""

import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_FILE = ROOT / "data" / "fixtures.json"
RATINGS_FILE = ROOT / "data" / "team_ratings.json"
OUTPUT_FILE = ROOT / "data" / "predictions.json"


def load_json(path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def norm(value):
    return " ".join(str(value or "").strip().lower().split())


def find_team_name(obj, candidates):
    for key in candidates:
        if key in obj and obj[key]:
            return str(obj[key])
    return ""


def extract_team(match, side):
    if side == "home":
        candidates = ["home_team", "homeTeam", "home", "home_name"]
        nested = ["homeTeam", "home"]
    else:
        candidates = ["away_team", "awayTeam", "away", "away_name"]
        nested = ["awayTeam", "away"]

    name = find_team_name(match, candidates)
    if name:
        return name

    for key in nested:
        value = match.get(key)
        if isinstance(value, dict):
            name = find_team_name(value, ["name", "team", "shortName"])
            if name:
                return name
    return ""


def extract_competition(match):
    for key in ["competition", "league", "competition_name", "tournament"]:
        value = match.get(key)
        if isinstance(value, dict):
            return value.get("name") or value.get("code") or ""
        if value:
            return str(value)
    return ""


def extract_id(match, index):
    for key in ["id", "fixture_id", "match_id"]:
        if match.get(key) is not None:
            return str(match[key])
    return f"fixture-{index + 1}"


def rating_for(team, ratings):
    if not team:
        return 1500.0
    key = norm(team)
    for name, value in ratings.items():
        if norm(name) == key:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return 1500.0


def softmax(values):
    exps = [math.exp(max(-20, min(20, x))) for x in values]
    total = sum(exps)
    return [x / total for x in exps]


def predict(home, away, ratings):
    """
    Conservative rating-based classifier.

    Elo-style difference is converted into three outcome scores.
    Home advantage is included. Draw receives a baseline probability.
    This is a starter model, not a validated production model.
    """
    home_rating = rating_for(home, ratings)
    away_rating = rating_for(away, ratings)

    diff = (home_rating + 55.0) - away_rating

    # Logistic win tendency.
    home_edge = 1.0 / (1.0 + 10 ** (-diff / 400.0))
    away_edge = 1.0 - home_edge

    # Draw baseline is reduced when teams are very mismatched.
    mismatch = min(abs(diff) / 600.0, 0.35)
    draw = 0.28 - (0.10 * mismatch)
    draw = max(0.16, min(0.28, draw))

    remaining = 1.0 - draw
    home_prob = home_edge * remaining
    away_prob = away_edge * remaining

    probs = softmax([
        math.log(max(home_prob, 1e-9)),
        math.log(max(draw, 1e-9)),
        math.log(max(away_prob, 1e-9)),
    ])

    return {
        "home_win": round(probs[0] * 100, 2),
        "draw": round(probs[1] * 100, 2),
        "away_win": round(probs[2] * 100, 2),
    }


def confidence_label(probs):
    ordered = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    top = ordered[0][1]
    gap = top - ordered[1][1]

    if top >= 65 and gap >= 18:
        return "HIGH"
    if top >= 55 and gap >= 10:
        return "MEDIUM"
    return "LOW"


def main():
    fixtures = load_json(FIXTURES_FILE, [])
    ratings = load_json(RATINGS_FILE, {})

    # Some APIs wrap fixtures in an object.
    if isinstance(fixtures, dict):
        for key in ["fixtures", "matches", "data"]:
            if isinstance(fixtures.get(key), list):
                fixtures = fixtures[key]
                break

    if not isinstance(fixtures, list):
        raise ValueError("data/fixtures.json must contain a list of fixtures.")

    predictions = []

    for index, match in enumerate(fixtures):
        if not isinstance(match, dict):
            continue

        home = extract_team(match, "home")
        away = extract_team(match, "away")

        if not home or not away:
            continue

        probs = predict(home, away, ratings)
        labels = {
            "home_win": "HOME WIN",
            "draw": "DRAW",
            "away_win": "AWAY WIN",
        }
        pick_key = max(probs, key=probs.get)

        predictions.append({
            "fixture_id": extract_id(match, index),
            "date": match.get("date") or match.get("utcDate") or match.get("match_date"),
            "competition": extract_competition(match),
            "home_team": home,
            "away_team": away,
            "probabilities": probs,
            "prediction": labels[pick_key],
            "confidence": confidence_label(probs),
            "model_version": "asta-v1-rating-baseline",
            "validated_accuracy": None,
            "status": "PENDING_RESULT",
        })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "asta-v1-rating-baseline",
        "accuracy_status": "NOT_VALIDATED",
        "accuracy_note": "Accuracy will only be reported after historical back-testing.",
        "total_predictions": len(predictions),
        "predictions": predictions,
    }

    OUTPUT_FILE.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Generated {len(predictions)} predictions.")
    print("Model status: baseline - NOT VALIDATED")
    print(f"Wrote predictions to {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
