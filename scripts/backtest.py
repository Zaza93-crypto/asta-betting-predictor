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

import json
import math
import sys
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
    "elo_diff",
    "home_advantage",
    "form5_diff",
    "form10_diff",
    "gf5_diff",
    "ga5_diff",
    "gd5_diff",
    "gf10_diff",
    "ga10_diff",
    "gd10_diff",
    "home_form5",
    "away_form5",
    "home_gf5",
    "away_gf5",
    "home_ga5",
    "away_ga5",
    "home_gd5",
    "away_gd5",
    "home_matches",
    "away_matches",
]


def load_matches():
    """Load and validate historical match data."""

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Historical data file not found: {INPUT}"
        )

    payload = json.loads(INPUT.read_text(encoding="utf-8"))

    matches = payload.get(
        "matches",
        payload if isinstance(payload, list) else []
    )

    out = []

    for m in matches:
        score = m.get("score") or {}

        hg = score.get("home")
        ag = score.get("away")

        home_team = m.get("home_team")
        away_team = m.get("away_team")

        if (
            home_team
            and away_team
            and isinstance(hg, int)
            and isinstance(ag, int)
        ):
            out.append(
                {
                    "date": m.get("date") or "",
                    "home_team": str(home_team),
                    "away_team": str(away_team),
                    "home_goals": hg,
                    "away_goals": ag,
                    "competition": str(
                        m.get("competition") or "UNKNOWN"
                    ),
                }
            )

    out.sort(key=lambda x: x["date"])

    return out


def outcome(home_goals, away_goals):
    """
    0 = Home win
    1 = Draw
    2 = Away win
    """

    if home_goals > away_goals:
        return 0

    if home_goals == away_goals:
        return 1

    return 2


def points(goals_for, goals_against):
    """Return football league points."""

    if goals_for > goals_against:
        return 3

    if goals_for == goals_against:
        return 1

    return 0


def avg(seq, idx, default=0.0):
    """Safely calculate an average from tuple data."""

    if not seq:
        return default

    return sum(x[idx] for x in seq) / len(seq)


def build_state():
    """
    Build chronological team state.

    Every history tuple is:
        (points, goals_for, goals_against)
    """

    return {
        "elo": defaultdict(lambda: 1500.0),

        "recent": defaultdict(
            lambda: deque(maxlen=10)
        ),

        "home": defaultdict(
            lambda: deque(maxlen=5)
        ),

        "away": defaultdict(
            lambda: deque(maxlen=5)
        ),

        "seen": defaultdict(int),
    }


def team_features(state, team, venue):
    """
    Generate features using only information available
    before the current match.
    """

    recent = list(state["recent"][team])
    venue_history = list(state[venue][team])

    return {
        "elo": state["elo"][team],

        "form5": avg(
            recent[-5:],
            0,
            1.0
        ),

        "form10": avg(
            recent[-10:],
            0,
            1.0
        ),

        "gf5": avg(
            recent[-5:],
            1,
            1.2
        ),

        "ga5": avg(
            recent[-5:],
            2,
            1.2
        ),

        "gd5": (
            avg(recent[-5:], 1, 1.2)
            - avg(recent[-5:], 2, 1.2)
        ),

        "gf10": avg(
            recent[-10:],
            1,
            1.2
        ),

        "ga10": avg(
            recent[-10:],
            2,
            1.2
        ),

        "gd10": (
            avg(recent[-10:], 1, 1.2)
            - avg(recent[-10:], 2, 1.2)
        ),

        "venue_form5": avg(
            venue_history[-5:],
            0,
            1.0
        ),

        "venue_gf5": avg(
            venue_history[-5:],
            1,
            1.2
        ),

        "venue_ga5": avg(
            venue_history[-5:],
            2,
            1.2
        ),

        "venue_gd5": (
            avg(venue_history[-5:], 1, 1.2)
            - avg(venue_history[-5:], 2, 1.2)
        ),

        "matches": state["seen"][team],
    }


def vector(home, away):
    """
    Convert team features into the V3 machine-learning vector.

    The venue features are kept as separate home/away values
    rather than duplicated differences.
    """

    return [
        # Elo and home advantage
        (home["elo"] - away["elo"]) / 400.0,
        1.0,

        # Overall recent form
        home["form5"] - away["form5"],
        home["form10"] - away["form10"],

        # Overall goals
        home["gf5"] - away["gf5"],
        home["ga5"] - away["ga5"],
        home["gd5"] - away["gd5"],

        home["gf10"] - away["gf10"],
        home["ga10"] - away["ga10"],
        home["gd10"] - away["gd10"],

        # Venue-specific performance
        home["venue_form5"],
        away["venue_form5"],

        home["venue_gf5"],
        away["venue_gf5"],

        home["venue_ga5"],
        away["venue_ga5"],

        home["venue_gd5"],
        away["venue_gd5"],

        # Experience
        math.log1p(home["matches"]),
        math.log1p(away["matches"]),
    ]


def update(state, match):
    """
    Update team state AFTER a match has been predicted.

    This ordering is important because it prevents
    look-ahead leakage.
    """

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_goals = match["home_goals"]
    away_goals = match["away_goals"]

    home_points = points(
        home_goals,
        away_goals
    )

    away_points = points(
        away_goals,
        home_goals
    )

    # Overall recent history
    state["recent"][home_team].append(
        (
            home_points,
            home_goals,
            away_goals
        )
    )

    state["recent"][away_team].append(
        (
            away_points,
            away_goals,
            home_goals
        )
    )

    # Home-specific history
    state["home"][home_team].append(
        (
            home_points,
            home_goals,
            away_goals
        )
    )

    # Away-specific history
    #
    # IMPORTANT:
    # This must contain THREE values:
    # (points, goals_for, goals_against)
    state["away"][away_team].append(
        (
            away_points,
            away_goals,
            home_goals
        )
    )

    # Match experience
    state["seen"][home_team] += 1
    state["seen"][away_team] += 1

    # Elo update
    expected = 1 / (
        1
        + 10 ** (
            -(
                (
                    state["elo"][home_team]
                    + 55
                )
                - state["elo"][away_team]
            )
            / 400
        )
    )

    actual = (
        1.0
        if home_goals > away_goals
        else 0.5
        if home_goals == away_goals
        else 0.0
    )

    change = 20 * (actual - expected)

    state["elo"][home_team] += change
    state["elo"][away_team] -= change


def fit_model(X, y):
    """
    Train the V3 logistic regression model.
    """

    model = Pipeline(
        [
            (
                "scale",
                StandardScaler()
            ),

            (
                "clf",
                LogisticRegression(
                    C=0.25,
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42
                )
            ),
        ]
    )

    model.fit(X, y)

    return model


def main():

    print("Loading historical matches...")

    matches = load_matches()

    print(
        f"Loaded {len(matches)} valid historical matches."
    )

    state = build_state()

    X_train = []
    y_train = []

    tested = 0
    correct = 0

    brier_sum = 0.0
    logloss_sum = 0.0

    # Minimum historical observations before
    # walk-forward predictions begin.
    warmup = 250

    by_comp = defaultdict(
        lambda: {
            "matches": 0,
            "correct": 0
        }
    )

    by_conf = defaultdict(
        lambda: {
            "matches": 0,
            "correct": 0
        }
    )

    for match in matches:

        home_features = team_features(
            state,
            match["home_team"],
            "home"
        )

        away_features = team_features(
            state,
            match["away_team"],
            "away"
        )

        x = vector(
            home_features,
            away_features
        )

        y = outcome(
            match["home_goals"],
            match["away_goals"]
        )

        # --------------------------------------------------
        # WALK-FORWARD PREDICTION
        # --------------------------------------------------
        #
        # Only matches BEFORE this match are available
        # to the model.
        #
        if (
            len(y_train) >= warmup
            and len(set(y_train)) == 3
        ):

            model = fit_model(
                X_train,
                y_train
            )

            probabilities = model.predict_proba(
                [x]
            )[0]

            classes = list(
                model.named_steps["clf"].classes_
            )

            probs = {
                0: 0.0,
                1: 0.0,
                2: 0.0
            }

            for class_value, probability in zip(
                classes,
                probabilities
            ):
                probs[int(class_value)] = float(
                    probability
                )

            prediction = max(
                probs,
                key=probs.get
            )

            is_correct = (
                prediction == y
            )

            tested += 1
            correct += int(is_correct)

            # --------------------------------------------------
            # BRIER SCORE
            # --------------------------------------------------

            for class_value in [0, 1, 2]:

                target = (
                    1.0
                    if class_value == y
                    else 0.0
                )

                brier_sum += (
                    probs[class_value]
                    - target
                ) ** 2

            # --------------------------------------------------
            # LOG LOSS
            # --------------------------------------------------

            logloss_sum -= math.log(
                max(
                    probs[y],
                    1e-12
                )
            )

            # --------------------------------------------------
            # CONFIDENCE BAND
            # --------------------------------------------------

            top_probability = max(
                probs.values()
            )

            if top_probability >= 0.65:

                confidence_band = (
                    "HIGH_65_PLUS"
                )

            elif top_probability >= 0.55:

                confidence_band = (
                    "MEDIUM_55_64"
                )

            else:

                confidence_band = (
                    "LOW_UNDER_55"
                )

            by_conf[
                confidence_band
            ]["matches"] += 1

            by_conf[
                confidence_band
            ]["correct"] += int(
                is_correct
            )

            # --------------------------------------------------
            # COMPETITION RESULTS
            # --------------------------------------------------

            competition = (
                match["competition"]
            )

            by_comp[
                competition
            ]["matches"] += 1

            by_comp[
                competition
            ]["correct"] += int(
                is_correct
            )

        # --------------------------------------------------
        # ADD CURRENT MATCH AFTER PREDICTION
        # --------------------------------------------------
        #
        # This is critical for preventing look-ahead leakage.
        #

        X_train.append(x)
        y_train.append(y)

        update(
            state,
            match
        )

    # ------------------------------------------------------
    # BUILD FINAL REPORT
    # ------------------------------------------------------

    report = {

        "generated_at":
            datetime.utcnow().isoformat() + "Z",

        "model_version":
            "asta-v3-final-walkforward-logistic",

        "status":
            "BACKTESTED"
            if tested
            else "NO_TESTABLE_MATCHES",

        "design": {

            "method":
                "expanding-window walk-forward logistic regression",

            "warmup_matches":
                warmup,

            "leakage_control":
                "Each match is predicted before its result enters training state.",

            "features":
                FEATURE_NAMES,

            "regularization":
                "C=0.25",

            "class_weight":
                "balanced"
        },

        "previous_versions": {

            "v1_accuracy":
                0.4874,

            "v2_accuracy":
                0.4829,

            "v1_matches_tested":
                1752,

            "v2_matches_tested":
                1752
        },

        "metrics": {

            "matches_tested":
                tested,

            "correct_predictions":
                correct,

            "accuracy":
                round(
                    correct / tested,
                    4
                )
                if tested
                else None,

            "brier_score":
                round(
                    brier_sum / tested,
                    6
                )
                if tested
                else None,

            "multiclass_log_loss":
                round(
                    logloss_sum / tested,
                    6
                )
                if tested
                else None
        },

        "by_competition": {

            competition: {
                **values,

                "accuracy":
                    round(
                        values["correct"]
                        / values["matches"],
                        4
                    )
            }

            for competition, values
            in by_comp.items()

            if values["matches"]
        },

        "by_confidence": {

            confidence: {
                **values,

                "accuracy":
                    round(
                        values["correct"]
                        / values["matches"],
                        4
                    )
            }

            for confidence, values
            in by_conf.items()

            if values["matches"]
        },

        "decision_rule":
            "Do not claim superiority unless V3 beats the relevant baseline on an appropriate out-of-sample sample."
    }

    # ------------------------------------------------------
    # SAVE REPORT
    # ------------------------------------------------------

    OUTPUT.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(
        f"SUCCESS: V3 tested {tested} chronological matches."
    )

    if tested:

        print(
            f"V3 accuracy: "
            f"{correct / tested * 100:.2f}%"
        )

        print(
            f"Brier: "
            f"{brier_sum / tested:.6f}"
        )

        print(
            f"Log loss: "
            f"{logloss_sum / tested:.6f}"
        )

    print(
        "Asta V3 is the final planned model version; "
        "future changes should be data/bug fixes, "
        "not version inflation."
    )


if __name__ == "__main__":
    main()
