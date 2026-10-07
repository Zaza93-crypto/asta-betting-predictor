#!/usr/bin/env python3
"""
Asta v2 chronological backtest.

Features are calculated only from matches that occurred BEFORE the match
being predicted. This is the key anti-leakage rule.
"""

import json, math
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "historical_matches.json"
OUTPUT = ROOT / "data" / "backtest_report.json"


def result(h, a):
    return "home_win" if h > a else "draw" if h == a else "away_win"


def avg(xs, i, default):
    return sum(x[i] for x in xs) / len(xs) if xs else default


def predict(hf, af):
    elo_diff = ((hf["elo"] + 55) - af["elo"]) / 400
    attack = .45*(hf["gf"]-af["ga"]) + .35*(hf["vgf"]-af["vga"])
    defense = .35*(af["gf"]-hf["ga"]) + .25*(af["vgf"]-hf["vga"])
    form = .22*(hf["points"]-af["points"]) + .18*(hf["vpoints"]-af["vpoints"])

    lam_h = max(.20, min(3.80, .95 + .75*elo_diff + .18*attack + .12*form))
    lam_a = max(.20, min(3.80, .95 - .75*elo_diff + .18*defense - .12*form))

    p = {"home_win":0.0, "draw":0.0, "away_win":0.0}
    for hg in range(7):
        ph = math.exp(-lam_h)*lam_h**hg/math.factorial(hg)
        for ag in range(7):
            pa = math.exp(-lam_a)*lam_a**ag/math.factorial(ag)
            if hg > ag: p["home_win"] += ph*pa
            elif hg == ag: p["draw"] += ph*pa
            else: p["away_win"] += ph*pa
    s = sum(p.values())
    return {k:v/s for k,v in p.items()}


def main():
    payload = json.loads(INPUT.read_text(encoding="utf-8"))
    matches = sorted(payload.get("matches", []), key=lambda x:x.get("date") or "")

    recent = defaultdict(lambda:deque(maxlen=10))
    homes = defaultdict(lambda:deque(maxlen=8))
    aways = defaultdict(lambda:deque(maxlen=8))
    elo = defaultdict(lambda:1500.0)

    correct = 0
    brier = 0.0
    logloss = 0.0
    tested = 0
    by_comp = defaultdict(lambda:{"matches":0,"correct":0})
    by_conf = defaultdict(lambda:{"matches":0,"correct":0})

    for m in matches:
        s=m.get("score") or {}
        hg,ag=s.get("home"),s.get("away")
        h,a=m.get("home_team"),m.get("away_team")
        if not h or not a or not isinstance(hg,int) or not isinstance(ag,int):
            continue

        def feat(team, venue):
            r=list(recent[team])
            v=list(homes[team] if venue=="home" else aways[team])
            return {
                "elo":elo[team],
                "points":avg(r,0,1.0),"gf":avg(r,1,1.35),"ga":avg(r,2,1.35),
                "vpoints":avg(v,0,1.0),"vgf":avg(v,1,1.35),"vga":avg(v,2,1.35)
            }

        # CRITICAL: predict before adding the current match to any history.
        p=predict(feat(h,"home"),feat(a,"away"))
        actual=result(hg,ag)
        pred=max(p,key=p.get)
        ok=pred==actual
        tested+=1
        correct+=int(ok)

        for k in p:
            y=1.0 if k==actual else 0.0
            brier+=(p[k]-y)**2
        logloss-=math.log(max(p[actual],1e-12))

        top=max(p.values())
        band="HIGH_65_PLUS" if top>=.65 else "MEDIUM_55_64" if top>=.55 else "LOW_UNDER_55"
        comp=m.get("competition","UNKNOWN")
        by_comp[comp]["matches"]+=1; by_comp[comp]["correct"]+=int(ok)
        by_conf[band]["matches"]+=1; by_conf[band]["correct"]+=int(ok)

        # Update after prediction.
        hp=1 if hg>ag else 2 if hg==ag else 0
        ap=1 if ag>hg else 2 if hg==ag else 0
        recent[h].append((hp,hg,ag)); recent[a].append((ap,ag,hg))
        homes[h].append((hp,hg,ag)); aways[a].append((ap,ag,hg))

        expected=1/(1+10**(-((elo[h]+55)-elo[a])/400))
        actual_home=1 if hg>ag else .5 if hg==ag else 0
        change=20*(actual_home-expected)
        elo[h]+=change; elo[a]-=change

    report={
        "generated_at":datetime.utcnow().isoformat()+"Z",
        "model_version":"asta-v2-form-elo-poisson",
        "status":"BACKTESTED" if tested else "NO_TESTABLE_MATCHES",
        "baseline_comparison":{
            "v1_accuracy":0.4874,
            "v1_matches_tested":1752,
            "comparison_note":"V2 should be compared on the same historical sample when possible."
        },
        "metrics":{
            "matches_tested":tested,
            "correct_predictions":correct,
            "accuracy":round(correct/tested,4) if tested else None,
            "brier_score":round(brier/tested,6) if tested else None,
            "multiclass_log_loss":round(logloss/tested,6) if tested else None
        },
        "by_competition":{
            k:{**v,"accuracy":round(v["correct"]/v["matches"],4)}
            for k,v in by_comp.items() if v["matches"]
        },
        "by_confidence":{
            k:{**v,"accuracy":round(v["correct"]/v["matches"],4)}
            for k,v in by_conf.items() if v["matches"]
        }
    }
    OUTPUT.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"SUCCESS: V2 back-tested {tested} matches.")
    print(f"Accuracy: {correct/tested*100:.2f}%" if tested else "No testable matches.")
    print("IMPORTANT: Historical results do not guarantee future accuracy or profitability.")


if __name__=="__main__":
    main()
