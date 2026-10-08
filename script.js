const DATA_URL = "data/predictions.json";

const fmtTime = (value) => {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
};

const bandLabel = (band) => ({
  HIGH_65_PLUS: "HIGH",
  MEDIUM_55_64: "MEDIUM",
  LOW_UNDER_55: "LOW"
}[band] || "—");

function card(m) {
  return `<article class="card">
    <div class="card-top"><span>${m.competition || "Football"}</span><span>${fmtTime(m.date)}</span></div>
    <div class="teams"><div class="team">${m.home_team}</div><span class="vs">VS</span><div class="team">${m.away_team}</div></div>
    <div class="pickline"><span class="pick">${m.prediction}</span><span class="confidence">${m.confidence_percent}% · ${bandLabel(m.confidence_band)}</span></div>
  </article>`;
}

function row(m) {
  const decision = m.decision || "—";
  return `<article class="match">
    <div><span class="muted">${m.competition || "Football"}</span><br><small>${fmtTime(m.date)}</small></div>
    <div class="teams"><div class="team">${m.home_team}</div><span class="vs">VS</span><div class="team">${m.away_team}</div></div>
    <div class="pick">${m.prediction} · ${m.confidence_percent}% · ${decision}</div>
  </article>`;
}

function setBand(id, band) {
  const row = band || {};
  document.getElementById(`${id}Matches`).textContent = row.matches ?? 0;
  document.getElementById(`${id}Accuracy`).textContent = row.accuracy == null ? "—" : `${row.accuracy}%`;
}

function show(data) {
  const predictions = Array.isArray(data.predictions) ? data.predictions : [];
  const primary = predictions.filter(x => x.decision === "BET_CANDIDATE");
  const r = data.performance || {tracked:0, correct:0, accuracy:null, confidence_bands:{}};
  const bands = r.confidence_bands || {};
  const decisions = data.decision_counts || {};

  document.getElementById("dataStatus").textContent = data.accuracy_status === "LIVE_TRACKING" ? "TRACKING" : "READY";
  document.getElementById("liveCount").textContent = predictions.length;
  document.getElementById("updated").textContent = "Updated " + (data.generated_at ? new Date(data.generated_at).toLocaleString() : "recently");
  document.getElementById("pickCount").textContent = r.tracked || 0;
  document.getElementById("correctCount").textContent = r.correct || 0;
  document.getElementById("accuracy").textContent = r.accuracy == null ? "—" : `${r.accuracy}%`;
  document.getElementById("modelVersion").textContent = data.model_version || "V3";

  document.getElementById("picksGrid").innerHTML = primary.slice(0,6).map(card).join("") || '<div class="loading">No HIGH confidence picks right now.</div>';
  document.getElementById("matchesGrid").innerHTML = predictions.slice(0,20).map(row).join("") || '<div class="loading">No predictions found.</div>';

  document.getElementById("performanceAccuracy").textContent = r.accuracy == null ? "—" : `${r.accuracy}%`;
  document.getElementById("performanceText").textContent = r.tracked ? `${r.correct} correct from ${r.tracked} verified predictions.` : "No verified live predictions yet.";
  document.getElementById("accuracyBar").style.width = `${Math.min(100, Math.max(0, r.accuracy || 0))}%`;

  document.getElementById("highDecision").textContent = decisions.BET_CANDIDATE ?? primary.length;
  document.getElementById("watchDecision").textContent = decisions.WATCH ?? 0;
  document.getElementById("skipDecision").textContent = decisions.SKIP ?? 0;
  setBand("high", bands.HIGH_65_PLUS);
  setBand("medium", bands.MEDIUM_55_64);
  setBand("low", bands.LOW_UNDER_55);
}

fetch(DATA_URL + "?v=" + Date.now())
  .then(response => { if (!response.ok) throw new Error("Prediction data unavailable"); return response.json(); })
  .then(show)
  .catch(() => show({accuracy_status:"NOT_READY", predictions:[], performance:{tracked:0,correct:0,accuracy:null,confidence_bands:{}}, decision_counts:{}}));

document.querySelector(".menu-btn").onclick = () => document.querySelector("nav").classList.toggle("open");
