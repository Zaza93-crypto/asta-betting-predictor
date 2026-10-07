import json, os, urllib.request, datetime
from pathlib import Path

KEY=os.environ.get("API_FOOTBALL_KEY")
OUT=Path("data/fixtures.json")
if not KEY:
    raise SystemExit("Missing API_FOOTBALL_KEY GitHub secret")

leagues={39:"Premier League",140:"La Liga",135:"Serie A",78:"Bundesliga",61:"Ligue 1"}
season=2026
headers={"x-apisports-key":KEY}
today=datetime.date.today()
date_from=today.isoformat()
date_to=(today+datetime.timedelta(days=2)).isoformat()

def get(url):
    req=urllib.request.Request(url,headers=headers)
    with urllib.request.urlopen(req,timeout=30) as r:
        return json.load(r)

matches=[]
for lid,lname in leagues.items():
    url=f"https://v3.football.api-sports.io/fixtures?league={lid}&season={season}&from={date_from}&to={date_to}"
    try:
        payload=get(url)
    except Exception:
        continue
    for item in payload.get("response",[]):
        fx=item["fixture"]; teams=item["teams"]
        if fx["status"]["short"] not in ("NS","TBD"):
            continue
        pred=None
        try:
            p=get(f"https://v3.football.api-sports.io/predictions?fixture={fx['id']}")
            pred=(p.get("response") or [{}])[0].get("predictions",{})
        except Exception:
            pred={}
        winner=(pred.get("winner") or {}).get("name")
        percent=pred.get("percent") or {}
        pick=winner or "No pick"
        conf=0
        if winner:
            vals=[]
            for v in percent.values():
                try: vals.append(float(str(v).replace("%","")))
                except: pass
            conf=round(max(vals)) if vals else 0
        matches.append({"id":fx["id"],"league":lname,"date":fx["date"],"home":teams["home"]["name"],"away":teams["away"]["name"],"pick":pick,"confidence":conf})

matches.sort(key=lambda x:x["date"])
data={"source":"api-football","updated":datetime.datetime.utcnow().replace(microsecond=0).isoformat()+"Z","matches":matches,"performance":{"tracked":0,"correct":0,"accuracy":None}}
OUT.parent.mkdir(exist_ok=True)
OUT.write_text(json.dumps(data,indent=2),encoding="utf-8")
print(f"Wrote {len(matches)} fixtures")
