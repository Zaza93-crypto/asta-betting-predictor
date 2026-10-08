#!/usr/bin/env python3
"""Resolve stored Asta V3 predictions against completed historical results."""

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACKING = ROOT / "data" / "prediction_history.json"
OUTPUT = ROOT / "data" / "predictions.json"
HISTORY = ROOT / "data" / "historical_matches.json"
MODEL_VERSION = "asta-v3-final-walkforward-logistic"
BANDS = ("HIGH_65_PLUS", "MEDIUM_55_64", "LOW_UNDER_55")


def label(home_goals, away_goals):
    return "HOME WIN" if home_goals > away_goals else "DRAW" if home_goals == away_goals else "AWAY WIN"


def date_key(value):
    return str(value or "")[:10]


def performance(records):
    verified = [row for row in records if row.get("result_status") == "VERIFIED"]
    correct = sum(row.get("correct") is True for row in verified)
    bands = {}
    for band in BANDS:
        rows = [row for row in verified if row.get("confidence_band") == band]
        hits = sum(row.get("correct") is True for row in rows)
        bands[band] = {
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


def load(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def main():
    tracking = load(TRACKING, {"predictions": []})
    records = tracking.get("predictions", []) if isinstance(tracking, dict) else []
    history = load(HISTORY, {})

    # Match on team names + UTC calendar date. We deliberately do not mark a
    # prediction verified from a fixture merely because the fixture disappeared.
    completed = {}
    for match in history.get("matches", []):
        score = match.get("score") or {}
        home = str(match.get("home_team") or "").strip()
        away = str(match.get("away_team") or "").strip()
        if not home or not away or not isinstance(score.get("home"), int) or not isinstance(score.get("away"), int):
            continue
        completed[(home, away, date_key(match.get("date")))] = (score["home"], score["away"])

    resolved = 0
    now = datetime.now(timezone.utc).isoformat()
    for record in records:
        if record.get("result_status") == "VERIFIED":
            continue
        key = (
            str(record.get("home_team") or "").strip(),
            str(record.get("away_team") or "").strip(),
            date_key(record.get("date")),
        )
        score = completed.get(key)
        if score is None:
            continue
        home_goals, away_goals = score
        actual = label(home_goals, away_goals)
        record.update({
            "actual_home_goals": home_goals,
            "actual_away_goals": away_goals,
            "actual_result": actual,
            "correct": actual == record.get("prediction"),
            "result_status": "VERIFIED",
            "resolved_at": now,
        })
        resolved += 1

    tracking.update({
        "model_version": MODEL_VERSION,
        "updated_at": now,
        "predictions": records,
        "performance": performance(records),
    })
    TRACKING.write_text(json.dumps(tracking, indent=2, ensure_ascii=False), encoding="utf-8")

    output = load(OUTPUT, {})
    output.update({
        "generated_at": now,
        "model_version": MODEL_VERSION,
        "accuracy_status": "LIVE_TRACKING",
        "performance": performance(records),
        "predictions": records,
    })
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Resolved {resolved} new predictions.")


if __name__ == "__main__":
    main()
