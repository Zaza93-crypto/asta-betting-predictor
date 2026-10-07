const DATA_URL="data/fixtures.json";
const fmtTime=(d)=>new Date(d).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
function card(m){return `<article class="card"><div class="card-top"><span>${m.league}</span><span>${fmtTime(m.date)}</span></div><div class="teams"><div class="team">${m.home}</div><span class="vs">VS</span><div class="team">${m.away}</div></div><div class="pickline"><span class="pick">${m.pick}</span><span class="confidence">${m.confidence}% confidence</span></div></article>`}
function row(m){return `<article class="match"><div><span class="muted">${m.league}</span><br><small>${fmtTime(m.date)}</small></div><div class="teams"><div class="team">${m.home}</div><span class="vs">VS</span><div class="team">${m.away}</div></div><div class="pick">${m.pick} · ${m.confidence}%</div></article>`}
function show(data){
 const matches=data.matches||[];
 document.getElementById("dataStatus").textContent=data.source==="api-football"?"LIVE":"DEMO";
 document.getElementById("liveCount").textContent=matches.length;
 document.getElementById("updated").textContent="Updated "+(data.updated||"recently");
 document.getElementById("picksGrid").innerHTML=matches.filter(x=>x.confidence>=60).slice(0,6).map(card).join("")||'<div class="loading">No qualifying picks yet.</div>';
 document.getElementById("matchesGrid").innerHTML=matches.slice(0,20).map(row).join("")||'<div class="loading">No upcoming matches found.</div>';
 const r=data.performance||{tracked:0,correct:0,accuracy:null};
 document.getElementById("pickCount").textContent=r.tracked;
 document.getElementById("correctCount").textContent=r.correct;
 document.getElementById("accuracy").textContent=r.accuracy==null?"—":r.accuracy+"%";
 document.getElementById("performanceAccuracy").textContent=r.accuracy==null?"—":r.accuracy+"%";
 document.getElementById("performanceText").textContent=r.tracked?`${r.correct} correct from ${r.tracked} verified predictions.`:"No verified historical predictions yet.";
 document.getElementById("accuracyBar").style.width=(r.accuracy||0)+"%";
}
fetch(DATA_URL+"?v="+Date.now()).then(r=>{if(!r.ok)throw Error();return r.json()}).then(show).catch(()=>show({source:"demo",updated:"demo dataset",matches:[
{league:"Premier League",date:"2026-10-07T19:30:00Z",home:"Arsenal",away:"Chelsea",pick:"1X",confidence:72},
{league:"La Liga",date:"2026-10-07T20:00:00Z",home:"Barcelona",away:"Valencia",pick:"Over 2.5",confidence:68},
{league:"Serie A",date:"2026-10-07T21:00:00Z",home:"Inter",away:"Napoli",pick:"BTTS",confidence:64}],performance:{tracked:0,correct:0,accuracy:null}}));
document.querySelector(".menu-btn").onclick=()=>document.querySelector("nav").classList.toggle("open");
