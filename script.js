const picks=[
 {league:"Premier League",time:"19:30",home:"Arsenal",away:"Chelsea",pick:"1X",confidence:72},
 {league:"La Liga",time:"20:00",home:"Barcelona",away:"Valencia",pick:"Over 2.5",confidence:68},
 {league:"Serie A",time:"21:00",home:"Inter",away:"Napoli",pick:"BTTS",confidence:64}
];
const matches=[
 ...picks,
 {league:"Bundesliga",time:"18:30",home:"Bayern",away:"Leverkusen",pick:"Over 2.5",confidence:61},
 {league:"Ligue 1",time:"20:45",home:"PSG",away:"Lyon",pick:"1",confidence:70},
 {league:"Premier League",time:"21:00",home:"Liverpool",away:"Everton",pick:"1X",confidence:67}
];
function pickCard(x){return `<article class="pick"><div class="pick-top"><span>${x.league}</span><span>${x.time}</span></div><div class="teams"><div class="team">${x.home}</div><span class="vs">VS</span><div class="team">${x.away}</div></div><div class="pick-bottom"><span class="prediction">${x.pick}</span><span class="confidence-small">${x.confidence}% confidence</span></div></article>`}
function matchRow(x){return `<article class="match"><div><span class="league">${x.league}</span><br><small>${x.time}</small></div><div class="teams"><div class="team">${x.home}</div><span class="vs">VS</span><div class="team">${x.away}</div></div><div class="prediction">${x.pick} · ${x.confidence}%</div></article>`}
document.getElementById("picksGrid").innerHTML=picks.map(pickCard).join("");
document.getElementById("matchesGrid").innerHTML=matches.map(matchRow).join("");
document.querySelector(".menu-btn").addEventListener("click",()=>document.querySelector("nav").classList.toggle("open"));
document.querySelectorAll("nav a").forEach(a=>a.addEventListener("click",()=>document.querySelector("nav").classList.remove("open")));
