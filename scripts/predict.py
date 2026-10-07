#!/usr/bin/env python3
"""
Asta V3 FINAL live predictor.

Trains on all available historical matches and predicts current fixtures.
Training uses only completed historical matches. Accuracy is NOT claimed here;
the backtest report is the authority for historical performance.
"""

import json, math, sys
from collections import defaultdict, deque
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

def outcome(hg, ag):
    return 0 if hg > ag else 1 if hg == ag else 2

def points(gf, ga):
    return 3 if gf > ga else 1 if gf == ga else 0

def avg(seq, idx, default=0.0):
    return sum(x[idx] for x in seq)/len(seq) if seq else default

def vector(h,a):
    return [
        (h["elo"]-a["elo"])/400, 1.0,
        h["form5"]-a["form5"], h["form10"]-a["form10"],
        h["gf5"]-a["gf5"], h["ga5"]-a["ga5"], h["gd5"]-a["gd5"],
        h["gf10"]-a["gf10"], h["ga10"]-a["ga10"], h["gd10"]-a["gd10"],
        h["venue_form5"]-a["venue_form5"], h["venue_form5"]-a["venue_form5"],
        h["venue_gf5"]-a["venue_gf5"], h["venue_gf5"]-a["venue_gf5"],
        h["venue_ga5"]-a["venue_ga5"], h["venue_ga5"]-a["venue_ga5"],
        h["venue_gd5"]-a["venue_gd5"], h["venue_gd5"]-a["venue_gd5"],
        math.log1p(h["matches"])-math.log1p(a["matches"]),
        math.log1p(a["matches"])-math.log1p(h["matches"])
    ]

def team_features(state, team, venue):
    r=list(state["recent"][team]); v=list(state[venue][team])
    return {
        "elo":state["elo"][team],
        "form5":avg(r[-5:],0,1),"form10":avg(r[-10:],0,1),
        "gf5":avg(r[-5:],1,1.2),"ga5":avg(r[-5:],2,1.2),
        "gd5":avg(r[-5:],1,1.2)-avg(r[-5:],2,1.2),
        "gf10":avg(r[-10:],1,1.2),"ga10":avg(r[-10:],2,1.2),
        "gd10":avg(r[-10:],1,1.2)-avg(r[-10:],2,1.2),
        "venue_form5":avg(v[-5:],0,1),"venue_gf5":avg(v[-5:],1,1.2),
        "venue_ga5":avg(v[-5:],2,1.2),
        "venue_gd5":avg(v[-5:],1,1.2)-avg(v[-5:],2,1.2),
        "matches":state["seen"][team]
    }

def update(state,m):
    h,a=m["home_team"],m["away_team"]; hg,ag=m["home_goals"],m["away_goals"]
    hp,ap=points(hg,ag),points(ag,hg)
    state["recent"][h].append((hp,hg,ag)); state["recent"][a].append((ap,ag,hg))
    state["home"][h].append((hp,hg,ag)); state["away"][a].append((ap,ag,hg))
    state["seen"][h]+=1; state["seen"][a]+=1
    expected=1/(1+10**(-((state["elo"][h]+55)-state["elo"][a])/400))
    actual=1 if hg>ag else .5 if hg==ag else 0
    change=20*(actual-expected)
    state["elo"][h]+=change; state["elo"][a]-=change

def main():
    hist=json.loads(HISTORY.read_text(encoding="utf-8"))
    matches=sorted(hist.get("matches",[]),key=lambda x:x.get("date") or "")
    state={
        "elo":defaultdict(lambda:1500.0),
        "recent":defaultdict(lambda:deque(maxlen=10)),
        "home":defaultdict(lambda:deque(maxlen=5)),
        "away":defaultdict(lambda:deque(maxlen=5)),
        "seen":defaultdict(int)
    }
    X=[]; y=[]
    for m in matches:
        h,a=m.get("home_team"),m.get("away_team"); s=m.get("score") or {}
        if not h or not a or not isinstance(s.get("home"),int) or not isinstance(s.get("away"),int):
            continue
        X.append(vector(team_features(state,h,"home"),team_features(state,a,"away")))
        y.append(outcome(s["home"],s["away"]))
        update(state,{"home_team":h,"away_team":a,"home_goals":s["home"],"away_goals":s["away"]})

    if len(X)<250 or len(set(y))<3:
        raise RuntimeError("Not enough historical data to train V3.")

    model=Pipeline([
        ("scale",StandardScaler()),
        ("clf",LogisticRegression(C=.25,max_iter=1500,class_weight="balanced",random_state=42))
    ])
    model.fit(X,y)

    fixtures=json.loads(FIXTURES.read_text(encoding="utf-8"))
    if isinstance(fixtures,dict):
        fixtures=fixtures.get("fixtures") or fixtures.get("matches") or fixtures.get("data") or []

    labels={0:"HOME WIN",1:"DRAW",2:"AWAY WIN"}
    predictions=[]
    for i,m in enumerate(fixtures):
        h=m.get("home_team") or m.get("homeTeam",{}).get("name") if isinstance(m.get("homeTeam"),dict) else m.get("home_team") or m.get("homeTeam")
        a=m.get("away_team") or m.get("awayTeam",{}).get("name") if isinstance(m.get("awayTeam"),dict) else m.get("away_team") or m.get("awayTeam")
        if not h or not a: continue
        p=model.predict_proba([vector(team_features(state,h,"home"),team_features(state,a,"away"))])[0]
        classes=list(model.named_steps["clf"].classes_)
        probs={int(c):0.0 for c in [0,1,2]}
        for c,val in zip(classes,p): probs[int(c)]=float(val)
        key=max(probs,key=probs.get)
        top=probs[key]; confidence="HIGH" if top>=.65 else "MEDIUM" if top>=.55 else "LOW"
        predictions.append({
            "fixture_id":str(m.get("id") or m.get("fixture_id") or f"fixture-{i+1}"),
            "date":m.get("date") or m.get("utcDate"),
            "competition":m.get("competition") or m.get("league") or "",
            "home_team":h,"away_team":a,
            "probabilities":{
                "home_win":round(probs[0]*100,2),
                "draw":round(probs[1]*100,2),
                "away_win":round(probs[2]*100,2)
            },
            "prediction":labels[key],
            "confidence":confidence,
            "model_version":"asta-v3-final-walkforward-logistic",
            "status":"PENDING_RESULT"
        })

    result={
        "generated_at":__import__("datetime").datetime.utcnow().isoformat()+"Z",
        "model_version":"asta-v3-final-walkforward-logistic",
        "accuracy_status":"NOT_VALIDATED",
        "accuracy_note":"Historical accuracy is reported only by data/backtest_report.json.",
        "total_predictions":len(predictions),
        "predictions":predictions
    }
    OUTPUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"SUCCESS: Generated {len(predictions)} V3 predictions.")

if __name__=="__main__":
    main()
