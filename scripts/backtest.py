#!/usr/bin/env python3
import json, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "historical_matches.json"
OUTPUT = ROOT / "data" / "backtest_report.json"

def result_key(h, a):
    return "home_win" if h > a else "draw" if h == a else "away_win"

def predict(hr, ar):
    diff = (hr + 55.0) - ar
    home_edge = 1 / (1 + 10 ** (-diff / 400.0))
    away_edge = 1 - home_edge
    mismatch = min(abs(diff) / 600.0, 0.35)
    draw = max(0.16, min(0.28, 0.28 - 0.10 * mismatch))
    remaining = 1 - draw
    home = home_edge * remaining
    away = away_edge * remaining
    total = home + draw + away
    return {"home_win": home/total, "draw": draw/total, "away_win": away/total}

def update_elo(hr, ar, actual, k=20.0):
    expected = 1 / (1 + 10 ** (-(hr + 55 - ar) / 400))
    actual_home = {"home_win": 1.0, "draw": 0.5, "away_win": 0.0}[actual]
    change = k * (actual_home - expected)
    return hr + change, ar - change

def main():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    matches = sorted(payload.get("matches", []), key=lambda x: x.get("date") or "")
    ratings = defaultdict(lambda: 1500.0)
    by_comp = defaultdict(lambda: {"matches": 0, "correct": 0})
    by_conf = defaultdict(lambda: {"matches": 0, "correct": 0})
    rows, correct, brier, logloss = [], 0, 0.0, 0.0

    for m in matches:
        s = m.get("score") or {}
        h, a = s.get("home"), s.get("away")
        home, away = m.get("home_team"), m.get("away_team")
        if not home or not away or not isinstance(h, int) or not isinstance(a, int):
            continue

        probs = predict(ratings[home], ratings[away])
        actual = result_key(h, a)
        pred = max(probs, key=probs.get)
        ok = pred == actual
        correct += int(ok)

        one_hot = {k: float(k == actual) for k in probs}
        brier += sum((probs[k] - one_hot[k]) ** 2 for k in probs)
        logloss += -math.log(max(probs[actual], 1e-12))

        top = max(probs.values()) * 100
        band = "HIGH_65_PLUS" if top >= 65 else "MEDIUM_55_64" if top >= 55 else "LOW_UNDER_55"
        comp = m.get("competition", "UNKNOWN")
        by_comp[comp]["matches"] += 1
        by_comp[comp]["correct"] += int(ok)
        by_conf[band]["matches"] += 1
        by_conf[band]["correct"] += int(ok)

        rows.append({
            "fixture_id": m.get("fixture_id"), "date": m.get("date"),
            "competition": comp, "home_team": home, "away_team": away,
            "probabilities": {k: round(v*100, 2) for k,v in probs.items()},
            "prediction": pred, "actual_result": actual, "correct": ok
        })

        ratings[home], ratings[away] = update_elo(ratings[home], ratings[away], actual)

    n = len(rows)
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": "asta-v1-sequential-elo-backtest",
        "status": "BACKTESTED" if n else "NO_TESTABLE_MATCHES",
        "metrics": {
            "matches_tested": n,
            "correct_predictions": correct,
            "accuracy": round(correct/n, 4) if n else None,
            "brier_score": round(brier/n, 6) if n else None,
            "multiclass_log_loss": round(logloss/n, 6) if n else None
        },
        "by_competition": {
            k: {**v, "accuracy": round(v["correct"]/v["matches"], 4)}
            for k,v in by_comp.items() if v["matches"]
        },
        "by_confidence": {
            k: {**v, "accuracy": round(v["correct"]/v["matches"], 4)}
            for k,v in by_conf.items() if v["matches"]
        },
        "sample_predictions": rows[:25]
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: Back-tested {n} matches.")
    print("IMPORTANT: Historical performance is not a guarantee of future accuracy.")

if __name__ == "__main__":
    main()
