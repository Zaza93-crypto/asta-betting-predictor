#!/usr/bin/env python3
"""
Asta Football Predictor - historical back-test.

Important:
- Matches are processed chronologically.
- Each prediction uses only ratings learned from matches BEFORE that match.
- The actual result is used to update ratings only AFTER the prediction.
- No future results are used to create a past prediction.

Metrics:
- 1X2 accuracy
- Brier score
- Multiclass log loss
- Results by competition
- Confidence-band accuracy
"""

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "historical_matches.json"
OUTPUT = ROOT / "data" / "backtest_report.json"


def load_matches():
    if not INPUT.exists():
        raise FileNotFoundError("data/historical_matches.json is missing. Run fetch_history.py first.")
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    matches = payload.get("matches", [])
    return sorted(matches, key=lambda x: x.get("date") or "")


def result_key(home_goals, away_goals):
    if home_goals > away_goals:
        return "home_win"
    if home_goals == away_goals:
        return "draw"
    return "away_win"


def predict(home_rating, away_rating, home_advantage=55.0):
    diff = (home_rating + home_advantage) - away_rating
    home_edge = 1 / (1 + 10 ** (-diff / 400))
    away_edge = 1 - home_edge

    mismatch = min(abs(diff) / 600, 0.35)
    draw = max(0.16, min(0.28, 0.28 - 0.10 * mismatch))
    remaining = 1 - draw

    home = home_edge * remaining
    away = away_edge * remaining

    total = home + draw + away
    return {
        "home_win": home / total,
        "draw": draw / total,
        "away_win": away / total,
    }


def update_elo(home_rating, away_rating, actual, k=20.0):
    expected_home = 1 / (1 + 10 ** (-(home_rating + 55 - away_rating) / 400))
    actual_home = {
        "home_win": 1.0,
        "draw": 0.5,
        "away_win": 0.0,
    }[actual]
    change = k * (actual_home - expected_home)
    return home_rating + change, away_rating - change


def safe_log(x):
    return math.log(max(min(x, 1 - 1e-12), 1e-12))


def main():
    matches = load_matches()

    ratings = defaultdict(lambda: 1500.0)
    rows = []
    correct = 0
    brier_total = 0.0
    logloss_total = 0.0
    by_comp = defaultdict(lambda: {"matches": 0, "correct": 0})
    confidence = defaultdict(lambda: {"matches": 0, "correct": 0})

    for match in matches:
        score = match.get("score") or {}
        hg = score.get("home")
        ag = score.get("away")
        home = match.get("home_team")
        away = match.get("away_team")

        # Only finished matches with a numeric full-time score are testable.
        if not home or not away or not isinstance(hg, int) or not isinstance(ag, int):
            continue

        probs = predict(ratings[home], ratings[away])
        actual = result_key(hg, ag)
        predicted = max(probs, key=probs.get)

        if predicted == actual:
            correct += 1

        one_hot = {k: 1.0 if k == actual else 0.0 for k in probs}
        brier_total += sum((probs[k] - one_hot[k]) ** 2 for k in probs)
        logloss_total += -safe_log(probs[actual])

        top_probability = max(probs.values()) * 100
        if top_probability >= 65:
            band = "HIGH_65_PLUS"
        elif top_probability >= 55:
            band = "MEDIUM_55_64"
        else:
            band = "LOW_UNDER_55"

        by_comp[match.get("competition", "UNKNOWN")]["matches"] += 1
        by_comp[match.get("competition", "UNKNOWN")]["correct"] += int(predicted == actual)
        confidence[band]["matches"] += 1
        confidence[band]["correct"] += int(predicted == actual)

        rows.append({
            "fixture_id": match.get("fixture_id"),
            "date": match.get("date"),
            "competition": match.get("competition"),
            "home_team": home,
            "away_team": away,
            "probabilities": {k: round(v * 100, 2) for k, v in probs.items()},
            "prediction": predicted,
            "actual_result": actual,
            "correct": predicted == actual,
            "pre_match_home_rating": round(ratings[home], 2),
            "pre_match_away_rating": round(ratings[away], 2),
        })

        # Update only AFTER the prediction has been evaluated.
        ratings[home], ratings[away] = update_elo(
            ratings[home], ratings[away], actual
        )

    n = len(rows)
    accuracy = correct / n if n else None
    brier = brier_total / n if n else None
    logloss = logloss_total / n if n else None

    competition_report = {}
    for comp, values in by_comp.items():
        competition_report[comp] = {
            "matches": values["matches"],
            "correct": values["correct"],
            "accuracy": round(values["correct"] / values["matches"], 4)
            if values["matches"] else None,
        }

    confidence_report = {}
    for band, values in confidence.items():
        confidence_report[band] = {
            "matches": values["matches"],
            "correct": values["correct"],
            "accuracy": round(values["correct"] / values["matches"], 4)
            if values["matches"] else None,
        }

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": "asta-v1-sequential-elo-backtest",
        "methodology": {
            "rating_start": 1500,
            "home_advantage": 55,
            "elo_k": 20,
            "data_leakage_protection": "Prediction is generated before the current match updates the ratings.",
        },
        "status": "BACKTESTED" if n else "NO_TESTABLE_MATCHES",
        "metrics": {
            "matches_tested": n,
            "correct_predictions": correct,
            "accuracy": round(accuracy, 4) if accuracy is not None else None,
            "brier_score": round(brier, 6) if brier is not None else None,
            "multiclass_log_loss": round(logloss, 6) if logloss is not None else None,
        },
        "by_competition": competition_report,
        "by_confidence": confidence_report,
        "sample_predictions": rows[:25],
    }

    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"SUCCESS: Back-tested {n} matches.")
    if accuracy is not None:
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Brier score: {brier:.6f}")
        print(f"Multiclass log loss: {logloss:.6f}")
    print("IMPORTANT: This is historical back-test performance, not a guarantee of future accuracy.")


if __name__ == "__main__":
    main()
