#!/usr/bin/env python3
"""
Asta V3 FINAL - walk-forward machine-learning backtest.

Design goals:
- No look-ahead leakage.
- Features for a match use only matches completed before that match.
- The classifier is retrained only on prior observations.
- Uses the same historical sample as the existing V1/V2 reports.
- Reports accuracy, Brier score, log loss, confidence-band accuracy,
  and a direct comparison with V1/V2.

This is a research/backtesting system, not a guarantee of future results.
"""

import json, math, sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("ERROR: scikit-learn is required. Run: pip install -r requirements.txt")
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "historical_matches.json"
OUTPUT = ROOT / "data" / "backtest_report.json"

FEATURE_NAMES = [
    "elo_diff", "home_advantage",
    "form5_diff", "form10_diff",
    "gf5_diff", "ga5_diff", "gd5_diff",
    "gf10_diff", "ga10_diff", "gd10_diff",
    "home_form5", "away_form5",
    "home_gf5", "away_gf5",
    "home_ga5", "away_ga5",
    "home_gd5", "away_gd5",
    "home_matches", "away_matches",
]


def load_matches():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    matches = payload.get("matches", payload if isinstance(payload, list) else [])
    out = []
    for m in matches:
        s = m.get("score") or {}
        hg, ag = s.get("home"), s.get("away")
        h, a = m.get("home_team"), m.get("away_team")
        if h and a and isinstance(hg, int) and isinstance(ag, int):
            out.append({
                "date": m.get("date") or "",
                "home_team": str(h),
                "away_team": str(a),
                "home_goals": hg,
                "away_goals": ag,
                "competition": str(m.get("competition") or "UNKNOWN"),
            })
    out.sort(key=lambda x: x["date"])
    return out


def outcome(hg, ag):
    return 0 if hg > ag else 1 if hg == ag else 2  # home, draw, away


def points(gf, ga):
    return 3 if gf > ga else 1 if gf == ga else 0


def avg(seq, idx, default=0.0):
    return sum(x[idx] for x in seq) / len(seq) if seq else default


def build_state():
    return {
        "elo": defaultdict(lambda: 1500.0),
        "recent": defaultdict(lambda: deque(maxlen=10)),
        "home": defaultdict(lambda: deque(maxlen=5)),
        "away": defaultdict(lambda: deque(maxlen=5)),
        "seen": defaultdict(int),
    }


def team_features(state, team, venue):
    r = list(state["recent"][team])
    v = list(state[venue][team])

    # tuples: points, gf, ga
    return {
        "elo": state["elo"][team],
        "form5": avg(r[-5:], 0, 1.0),
        "form10": avg(r[-10:], 0, 1.0),
        "gf5": avg(r[-5:], 1, 1.2),
        "ga5": avg(r[-5:], 2, 1.2),
        "gd5": avg(r[-5:], 1, 1.2) - avg(r[-5:], 2, 1.2),
        "gf10": avg(r[-10:], 1, 1.2),
        "ga10": avg(r[-10:], 2, 1.2),
        "gd10": avg(r[-10:], 1, 1.2) - avg(r[-10:], 2, 1.2),
        "venue_form5": avg(v[-5:], 0, 1.0),
        "venue_gf5": avg(v[-5:], 1, 1.2),
        "venue_ga5": avg(v[-5:], 2, 1.2),
        "venue_gd5": avg(v[-5:], 1, 1.2) - avg(v[-5:], 2, 1.2),
        "matches": state["seen"][team],
    }


def vector(h, a):
    return [
        (h["elo"] - a["elo"]) / 400.0,
        1.0,
        h["form5"] - a["form5"],
        h["form10"] - a["form10"],
        h["gf5"] - a["gf5"],
        h["ga5"] - a["ga5"],
        h["gd5"] - a["gd5"],
        h["gf10"] - a["gf10"],
        h["ga10"] - a["ga10"],
        h["gd10"] - a["gd10"],
        h["venue_form5"] - a["venue_form5"],
        h["venue_form5"] - a["venue_form5"],
        h["venue_gf5"] - a["venue_gf5"],
        h["venue_gf5"] - a["venue_gf5"],
        h["venue_ga5"] - a["venue_ga5"],
        h["venue_ga5"] - a["venue_ga5"],
        h["venue_gd5"] - a["venue_gd5"],
        h["venue_gd5"] - a["venue_gd5"],
        math.log1p(h["matches"]) - math.log1p(a["matches"]),
        math.log1p(a["matches"]) - math.log1p(h["matches"]),
    ]


def update(state, m):
    h, a = m["home_team"], m["away_team"]
    hg, ag = m["home_goals"], m["away_goals"]

    hp = points(hg, ag)
    ap = points(ag, hg)

    state["recent"][h].append((hp, hg, ag))
    state["recent"][a].append((ap, ag, hg))
    state["home"][h].append((hp, hg, ag))
    state["away"][a].append((ap, ag))
    state["seen"][h] += 1
    state["seen"][a] += 1

    expected = 1 / (1 + 10 ** (-((state["elo"][h] + 55) - state["elo"][a]) / 400))
    actual = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
    change = 20 * (actual - expected)
    state["elo"][h] += change
    state["elo"][a] -= change


def fit_model(X, y):
    # Strong regularization reduces the chance of fitting noise in the
    # relatively small historical sample.
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=0.25,
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ))
    ])
    model.fit(X, y)
    return model


def main():
    matches = load_matches()
    state = build_state()

    X_train, y_train = [], []
    tested = correct = 0
    brier_sum = 0.0
    logloss_sum = 0.0
    warmup = 250

    by_comp = defaultdict(lambda: {"matches": 0, "correct": 0})
    by_conf = defaultdict(lambda: {"matches": 0, "correct": 0})

    for m in matches:
        h = team_features(state, m["home_team"], "home")
        a = team_features(state, m["away_team"], "away")
        x = vector(h, a)
        y = outcome(m["home_goals"], m["away_goals"])

        # Predict using ONLY earlier matches.
        if len(y_train) >= warmup and len(set(y_train)) == 3:
            model = fit_model(X_train, y_train)
            p = model.predict_proba([x])[0]
            classes = list(model.named_steps["clf"].classes_)
            probs = {int(c): 0.0 for c in [0, 1, 2]}
            for c, value in zip(classes, p):
                probs[int(c)] = float(value)

            pred = max(probs, key=probs.get)
            ok = pred == y
            tested += 1
            correct += int(ok)

            for c in [0, 1, 2]:
                target = 1.0 if c == y else 0.0
                brier_sum += (probs[c] - target) ** 2
            logloss_sum -= math.log(max(probs[y], 1e-12))

            top = max(probs.values())
            if top >= 0.65:
                band = "HIGH_65_PLUS"
            elif top >= 0.55:
                band = "MEDIUM_55_64"
            else:
                band = "LOW_UNDER_55"
            by_conf[band]["matches"] += 1
            by_conf[band]["correct"] += int(ok)

            comp = m["competition"]
            by_comp[comp]["matches"] += 1
            by_comp[comp]["correct"] += int(ok)

        # Add this match to training history AFTER its prediction.
        X_train.append(x)
        y_train.append(y)
        update(state, m)

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model_version": "asta-v3-final-walkforward-logistic",
        "status": "BACKTESTED" if tested else "NO_TESTABLE_MATCHES",
        "design": {
            "method": "expanding-window walk-forward logistic regression",
            "warmup_matches": warmup,
            "leakage_control": "Each match is predicted before its result enters training state.",
            "features": FEATURE_NAMES,
            "regularization": "C=0.25",
            "class_weight": "balanced"
        },
        "previous_versions": {
            "v1_accuracy": 0.4874,
            "v2_accuracy": 0.4829,
            "v1_matches_tested": 1752,
            "v2_matches_tested": 1752
        },
        "metrics": {
            "matches_tested": tested,
            "correct_predictions": correct,
            "accuracy": round(correct / tested, 4) if tested else None,
            "brier_score": round(brier_sum / tested, 6) if tested else None,
            "multiclass_log_loss": round(logloss_sum / tested, 6) if tested else None
        },
        "by_competition": {
            k: {**v, "accuracy": round(v["correct"] / v["matches"], 4)}
            for k, v in by_comp.items() if v["matches"]
        },
        "by_confidence": {
            k: {**v, "accuracy": round(v["correct"] / v["matches"], 4)}
            for k, v in by_conf.items() if v["matches"]
        },
        "decision_rule": "Do not claim superiority unless V3 beats the relevant baseline on an appropriate out-of-sample sample."
    }

    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SUCCESS: V3 tested {tested} chronological matches.")
    if tested:
        print(f"V3 accuracy: {correct / tested * 100:.2f}%")
        print(f"Brier: {brier_sum / tested:.6f}")
        print(f"Log loss: {logloss_sum / tested:.6f}")
    print("Asta V3 is the final planned model version; future changes should be data/bug fixes, not version inflation.")


if __name__ == "__main__":
    main()
