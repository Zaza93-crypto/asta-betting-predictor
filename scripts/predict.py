#!/usr/bin/env python3
"""Asta V3 final live predictor.

V3 responsibilities:
- Train a 3-way logistic model on completed historical matches.
- Use the same feature layout as the V3 walk-forward backtest.
- Classify every prediction by confidence.
- Persist prediction records so completed results can be verified later.

Important: the 65% confidence threshold is a decision filter, not a guarantee.
Historical and live accuracy are reported separately.
"""

import json
import math
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required. Run: pip install -r requirements.txt")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "data" / "historical_matches.json"
FIXTURES = ROOT / "data" / "fixtures.json"
OUTPUT = ROOT / "data" / "predictions.json"
TRACKING = ROOT / "data" / "prediction_history.json"

MODEL_VERSION = "asta-v3-final-walkforward-logistic"
LABELS = {0: "HOME WIN", 1: "DRAW", 2: "AWAY WIN"}
BANDS = ("HIGH_65_PLUS", "MEDIUM_55_64", "LOW_UNDER_55")


def outcome(home_goals, away_goals):
    return 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2


def points(goals_for, goals_against):
    return 3 if goals_for > goals_against else 1 if goals_for == goals_against else 0


def avg(values, index, default=0.0):
    return sum(row[index] for row in values) / len(values) if values else default


def vector(home, away):
    """Exactly the 20-feature layout documented by the V3 backtest."""
    return [
        (home["elo"] - away["elo"]) / 400.0,
        1.0,
        home["form5"] - away["form5"],
        home["form10"] - away["form10"],
        home["gf5"] - away["gf5"],
        home["ga5"] - away["ga5"],
        home["gd5"] - away["gd5"],
        home["gf10"] - away["gf10"],
        home["ga10"] - away["ga10"],
        home["gd10"] - away["gd10"],
        home["venue_form5"],
        away["venue_form5"],
        home["venue_gf5"],
        away["venue_gf5"],
        home["venue_ga5"],
        away["venue_ga5"],
        home["venue_gd5"],
        away["venue_gd5"],
        math.log1p(home["matches"]),
        math.log1p(away["matches"]),
    ]


def build_state():
    return {
        "elo": defaultdict(lambda: 1500.0),
        "recent": defaultdict(lambda: deque(maxlen=10)),
        "home": defaultdict(lambda: deque(maxlen=5)),
        "away": defaultdict(lambda: deque(maxlen=5)),
        "seen": defaultdict(int),
    }


def team_features(state, team, venue):
    recent = list(state["recent"][team])
    venue_results = list(state[venue][team])
    return {
        "elo": state["elo"][team],
        "form5": avg(recent[-5:], 0, 1.0),
        "form10": avg(recent[-10:], 0, 1.0),
        "gf5": avg(recent[-5:], 1, 1.2),
        "ga5": avg(recent[-5:], 2, 1.2),
        "gd5": avg(recent[-5:], 1, 1.2) - avg(recent[-5:], 2, 1.2),
        "gf10": avg(recent[-10:], 1, 1.2),
        "ga10": avg(recent[-10:], 2, 1.2),
        "gd10": avg(recent[-10:], 1, 1.2) - avg(recent[-10:], 2, 1.2),
        "venue_form5": avg(venue_results[-5:], 0, 1.0),
        "venue_gf5": avg(venue_results[-5:], 1, 1.2),
        "venue_ga5": avg(venue_results[-5:], 2, 1.2),
        "venue_gd5": avg(venue_results[-5:], 1, 1.2) - avg(venue_results[-5:], 2, 1.2),
        "matches": state["seen"][team],
    }


def update_state(state, match):
    home = match["home_team"]
    away = match["away_team"]
    hg = match["home_goals"]
    ag = match["away_goals"]
    home_points = points(hg, ag)
    away_points = points(ag, hg)

    state["recent"][home].append((home_points, hg, ag))
    state["recent"][away].append((away_points, ag, hg))
    state["home"][home].append((home_points, hg, ag))
    state["away"][away].append((away_points, ag, hg))
    state["seen"][home] += 1
    state["seen"][away] += 1

    expected_home = 1.0 / (1.0 + 10 ** (-((state["elo"][home] + 55.0) - state["elo"][away]) / 400.0))
    actual_home = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    change = 20.0 * (actual_home - expected_home)
    state["elo"][home] += change
    state["elo"][away] -= change


def train_model(features, labels):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.25,
            max_iter=1500,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    model.fit(features, labels)
    return model


def fixture_team(match, side):
    """Support both the API fixture shape and the older home_team/away_team shape."""
    direct = match.get(f"{side}_team")
    if direct:
        return str(direct).strip()
    value = match.get(side)
    if isinstance(value, dict):
        return str(value.get("name") or "").strip()
    return str(value or "").strip()


def load_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read {path}: {exc}") from exc


def load_tracking():
    payload = load_json(TRACKING, {"predictions": []})
    rows = payload.get("predictions", []) if isinstance(payload, dict) else []
    return rows if isinstance(rows, list) else []


def confidence_band(probability):
    if probability >= 0.65:
        return "HIGH_65_PLUS"
    if probability >= 0.55:
        return "MEDIUM_55_64"
    return "LOW_UNDER_55"


def decision(probability):
    if probability >= 0.65:
        return "BET_CANDIDATE"
    if probability >= 0.55:
        return "WATCH"
    return "SKIP"


def performance(records):
    verified = [row for row in records if row.get("result_status") == "VERIFIED"]
    correct = sum(row.get("correct") is True for row in verified)
    bands = {}
    for band_name in BANDS:
        rows = [row for row in verified if row.get("confidence_band") == band_name]
        hits = sum(row.get("correct") is True for row in rows)
        bands[band_name] = {
            "matches": len(rows),
            "correct": hits,
            "accuracy": round(hits / len(rows) * 100.0, 2) if rows else None,
        }
    return {
        "tracked": len(verified),
        "correct": correct,
        "accuracy": round(correct / len(verified) * 100.0, 2) if verified else None,
        "confidence_bands": bands,
    }


def main():
    history = load_json(HISTORY, {})
    matches = sorted(history.get("matches", []), key=lambda row: row.get("date") or "")
    state = build_state()
    features = []
    labels = []

    for match in matches:
        home = match.get("home_team")
        away = match.get("away_team")
        score = match.get("score") or {}
        if not home or not away or not isinstance(score.get("home"), int) or not isinstance(score.get("away"), int):
            continue

        # Predict using state BEFORE this match, then add the result to state.
        features.append(vector(
            team_features(state, home, "home"),
            team_features(state, away, "away"),
        ))
        labels.append(outcome(score["home"], score["away"]))
        update_state(state, {
            "home_team": home,
            "away_team": away,
            "home_goals": score["home"],
            "away_goals": score["away"],
        })

    if len(features) < 250 or len(set(labels)) < 3:
        raise RuntimeError("Not enough historical data to train V3.")

    model = train_model(features, labels)
    fixture_payload = load_json(FIXTURES, {})
    fixtures = fixture_payload.get("matches", []) if isinstance(fixture_payload, dict) else fixture_payload
    if not isinstance(fixtures, list):
        fixtures = []

    existing = {str(row.get("fixture_id")): row for row in load_tracking() if row.get("fixture_id") is not None}
    current = []
    now = datetime.now(timezone.utc).isoformat()

    for index, fixture in enumerate(fixtures):
        home = fixture_team(fixture, "home")
        away = fixture_team(fixture, "away")
        if not home or not away:
            continue

        probabilities_array = model.predict_proba([
            vector(team_features(state, home, "home"), team_features(state, away, "away"))
        ])[0]
        classes = list(model.named_steps["clf"].classes_)
        probabilities = {0: 0.0, 1: 0.0, 2: 0.0}
        for class_id, value in zip(classes, probabilities_array):
            probabilities[int(class_id)] = float(value)

        predicted_class = max(probabilities, key=probabilities.get)
        top_probability = probabilities[predicted_class]
        fixture_id = str(fixture.get("id") or fixture.get("fixture_id") or f"fixture-{index + 1}")

        record = existing.get(fixture_id, {}).copy()
        record.update({
            "fixture_id": fixture_id,
            "date": fixture.get("date") or fixture.get("utcDate"),
            "competition": fixture.get("league") or fixture.get("competition") or fixture.get("competition_code") or "",
            "home_team": home,
            "away_team": away,
            "probabilities": {
                "home_win": round(probabilities[0] * 100.0, 2),
                "draw": round(probabilities[1] * 100.0, 2),
                "away_win": round(probabilities[2] * 100.0, 2),
            },
            "prediction": LABELS[predicted_class],
            "confidence_percent": round(top_probability * 100.0, 2),
            "confidence_band": confidence_band(top_probability),
            "decision": decision(top_probability),
            "eligible_for_primary_picks": top_probability >= 0.65,
            "model_version": MODEL_VERSION,
            "generated_at": now,
        })

        record.setdefault("result_status", "PENDING_RESULT")
        record.setdefault("actual_result", None)
        record.setdefault("actual_home_goals", None)
        record.setdefault("actual_away_goals", None)
        record.setdefault("correct", None)
        record.setdefault("resolved_at", None)
        current.append(record)
        existing[fixture_id] = record

    records = sorted(existing.values(), key=lambda row: (row.get("date") or "", row.get("fixture_id") or ""))
    tracking_payload = {
        "model_version": MODEL_VERSION,
        "updated_at": now,
        "predictions": records,
    }
    TRACKING.write_text(json.dumps(tracking_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    decision_counts = {
        name: sum(row.get("decision") == name for row in current)
        for name in ("BET_CANDIDATE", "WATCH", "SKIP")
    }
    result = {
        "generated_at": now,
        "model_version": MODEL_VERSION,
        "accuracy_status": "LIVE_TRACKING",
        "accuracy_note": "Live accuracy is reported only after prediction results are verified. Historical backtest metrics remain separate.",
        "total_predictions": len(current),
        "decision_counts": decision_counts,
        "performance": performance(records),
        "predictions": current,
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Generated {len(current)} V3 predictions.")
    print(f"DECISIONS: {decision_counts}")


if __name__ == "__main__":
    main()
