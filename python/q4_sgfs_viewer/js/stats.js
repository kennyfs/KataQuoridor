"use strict";
// ---- Statistics tab ----------------------------------------------------------------------
// Shown: games, plies (mean / median / histogram), winners by seat (wins; and a score where a draw is 1/4 for each
// seat), draws by reason and rule, the winner's mean win probability by ply, calibration, searches (full vs cheap,
// visits), comebacks, and for match files a table per player and per table.
// Skipped: per-opening breakdowns, per-model learning curves across files (one file at a time), blunder statistics.
const tile = (k, v, s = "") => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`;
const card = (title, desc, body, wide = false) => `<div class="card${wide ? " wide" : ""}"><h3>${title}</h3><div class="desc">${desc}</div>${body}</div>`;
const median = a => { if (!a.length) return 0; const b = a.slice().sort((x, y) => x - y), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;

function barRow(label, sub, parts, nums) {
  const bar = parts.map(p => `<div style="flex:${Math.max(p.v, 0)};background:${p.c}" title="${escapeHtml(p.t || "")}"></div>`).join("");
  return `<div class="wr-row"><div class="lab">${label}${sub ? `<small>${sub}</small>` : ""}</div><div><div class="wr-bar">${bar}</div><div class="wr-nums">${nums}</div></div></div>`;
}
function seatBars(wins, draws, finished, score) {
  const rows = [];
  const parts = [0, 1, 2, 3].map(s => ({ v: score ? wins[s] + draws / 4 : wins[s], c: SEAT_COL[s], t: SEATS[s] }));
  if (!score) parts.push({ v: draws, c: DRAW_COL, t: "draw" });
  const tot = finished || 1;
  const nums = [0, 1, 2, 3].map(s => `${SEATS[s]} ${pct((score ? wins[s] + draws / 4 : wins[s]) / tot, 1)}`).concat(score ? [] : [`draw ${pct(draws / tot, 1)}`]).join(" · ");
  return barRow(score ? "Score by seat" : "Wins by seat", score ? "win = 1, draw = ¼ for each seat (sums to 100%)" : "share of finished games; the draw share is separate", parts, nums);
}
function histogram(values, binW, maxV, color, labelFmt = x => x) {
  const bins = Math.max(1, Math.ceil((maxV + 1) / binW)), cnt = new Array(bins).fill(0);
  for (const v of values) cnt[Math.min(bins - 1, Math.floor(v / binW))]++;
  const W = 520, H = 150, L = 34, B = 22, T = 8, mx = Math.max(...cnt, 1), bw = (W - L - 6) / bins;
  let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
  for (const t of [0, 0.5, 1]) svg += `<line x1="${L}" x2="${W - 6}" y1="${T + (H - T - B) * (1 - t)}" y2="${T + (H - T - B) * (1 - t)}" stroke="#e8e4da"/><text x="${L - 4}" y="${T + (H - T - B) * (1 - t) + 3.5}" text-anchor="end" font-size="10" fill="#8a96a3">${Math.round(mx * t)}</text>`;
  cnt.forEach((c, i) => {
    const h = (H - T - B) * c / mx;
    svg += `<rect x="${L + i * bw + 1}" y="${H - B - h}" width="${Math.max(1, bw - 2)}" height="${h}" rx="2" fill="${color}"><title>${labelFmt(i * binW)}–${labelFmt((i + 1) * binW - 1)}: ${c}</title></rect>`;
    if (bins <= 14 || i % Math.ceil(bins / 12) === 0) svg += `<text x="${L + i * bw + bw / 2}" y="${H - 6}" text-anchor="middle" font-size="10" fill="#8a96a3">${labelFmt(i * binW)}</text>`;
  });
  return svg + "</svg>";
}
function lineChart(series, xMax, opts = {}) {
  const W = 520, H = 190, L = 36, R = 8, T = 8, B = 22;
  const X = x => L + (W - L - R) * x / Math.max(xMax, 1), Y = y => T + (H - T - B) * (1 - y);
  let svg = `<svg viewBox="0 0 ${W} ${H}">`;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) svg += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="#e8e4da"/><text x="${L - 4}" y="${Y(t) + 3.5}" text-anchor="end" font-size="10" fill="#8a96a3">${Math.round(t * 100)}%</text>`;
  const step = Math.max(10, Math.ceil(xMax / 8 / 10) * 10);
  for (let x = 0; x <= xMax; x += step) svg += `<text x="${X(x)}" y="${H - 6}" text-anchor="middle" font-size="10" fill="#8a96a3">${x}</text>`;
  if (opts.diag) svg += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(xMax)}" y2="${Y(1)}" stroke="#9aa5b1" stroke-dasharray="4 3"/>`;
  for (const s of series) {
    const pts = s.pts.filter(p => p[1] !== null);
    if (pts.length > 1) svg += `<path d="${pts.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join("")}" fill="none" stroke="${s.c}" stroke-width="${s.w || 2}"/>`;
    if (s.dots) for (const p of pts) svg += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="${Math.min(9, 3 + Math.sqrt((p[2] || 1) / (s.dotScale || 5)))}" fill="${s.c}" opacity=".75"><title>${s.title ? s.title(p) : ""}</title></circle>`;
  }
  return svg + "</svg>";
}

function renderStats() {
  const idx = listOrder, H = idx.map(i => heads[i]), S = idx.map(i => sums[i]);
  const n = idx.length, f = activeFilters();
  $("statsNote").textContent = `${n} of ${heads.length} games` + (sumDone < lines.length ? ` · summary pass ${sumDone}/${lines.length}: evaluation statistics are partial` : "");
  $("statsFilters").textContent = f.length ? "Filters: " + f.join(" · ") : "No filters";
  if (!n) { $("statTiles").innerHTML = ""; $("statCards").innerHTML = "<div class=note>No games match the filters.</div>"; $("matchSection").hidden = true; return; }

  const fin = H.filter(h => h.finished), wins = [0, 0, 0, 0];
  let draws = 0; const drawBy = {};
  for (const h of fin) { if (h.winner >= 0) wins[h.winner]++; else { draws++; drawBy[h.drawReason] = (drawBy[h.drawReason] || 0) + 1; } }
  const plies = H.map(h => h.plies), withElim = H.filter(h => h.elims > 0).length;
  const sEval = S.filter(s => s && s.hasEval);
  $("statTiles").innerHTML = tile("Games", n, `${fin.length} finished, ${n - fin.length} unfinished`)
    + tile("Plies", mean(plies).toFixed(1), `median ${median(plies)} · max ${Math.max(...plies)}`)
    + tile("Draws", pct(draws / (fin.length || 1), 1), `${draws} games`)
    + tile("Eliminations", pct(withElim / n, 1), `${withElim} game${withElim === 1 ? "" : "s"} with one`)
    + tile("With search values", sEval.length, `${S.filter(s => s).length} summarized`);

  const cards = [];
  cards.push(card("Winners by seat", "Wins and score; a draw is ¼ point for each seat in the score bar (as in the match score), and its share is shown separately above.",
    seatBars(wins, draws, fin.length, false) + seatBars(wins, draws, fin.length, true)));
  const reasons = Object.keys(drawBy).sort();
  cards.push(card("Draws by reason", `${draws} draws of ${fin.length} finished games`,
    reasons.length ? `<table class="st"><tr><th>reason</th><th>games</th><th>share of finished</th></tr>${reasons.map(r => `<tr><td>${escapeHtml(r)}</td><td>${drawBy[r]}</td><td>${pct(drawBy[r] / fin.length, 1)}</td></tr>`).join("")}</table>` : "<div class=note>No draws.</div>"));
  const rules = [...new Set(H.map(h => h.rep))].sort((a, b) => a - b);
  cards.push(card("Results by repetition rule", "Rule off = repetitionDrawCount 0",
    `<table class="st"><tr><th>rule</th><th>games</th><th>S</th><th>W</th><th>N</th><th>E</th><th>draw</th><th>mean plies</th></tr>${rules.map(r => {
      const g = H.filter(h => h.rep === r), f2 = g.filter(h => h.finished), w = [0, 1, 2, 3].map(s => f2.filter(h => h.winner === s).length), d = f2.filter(h => h.isDraw).length;
      return `<tr><td>${r ? "rep " + r : "off"}</td><td>${g.length}</td>${w.map(x => `<td>${pct(x / (f2.length || 1))}</td>`).join("")}<td>${pct(d / (f2.length || 1))}</td><td>${mean(g.map(h => h.plies)).toFixed(1)}</td></tr>`;
    }).join("")}</table>`));
  const maxPl = Math.max(...plies);
  cards.push(card("Length of the games", "Histogram of the plies (moves, eliminations not counted)", histogram(plies, Math.max(5, Math.ceil(maxPl / 20 / 5) * 5), maxPl, "var(--board)")));

  // evaluation statistics
  if (sEval.length) {
    const wp = [];
    for (const s of S) if (s) for (const [pl, p] of s.winnerProb) { (wp[pl] = wp[pl] || []).push(p); }
    const pts = wp.map((a, i) => a && a.length >= 3 ? [i, mean(a), a.length] : [i, null]);
    cards.push(card("Does the model find out early?", "Mean win probability of the eventual winner at each ply (decided games, bins with at least 3 samples)", lineChart([{ pts, c: "var(--board)", w: 2.2 }], wp.length - 1)));
    // calibration: the seat to move at every 10th ply, a draw counts 1/4
    const bins = Array.from({ length: 10 }, () => ({ p: 0, o: 0, n: 0 }));
    for (const s of S) if (s) for (const [p, o] of s.calib) { const b = bins[Math.min(9, Math.floor(p * 10))]; b.p += p; b.o += o; b.n++; }
    const cp = bins.filter(b => b.n).map(b => [b.p / b.n, b.o / b.n, b.n]);
    cards.push(card("Calibration", "Predicted win probability of the seat to move (every 10th ply) vs its realised score (draw = ¼); dot size = number of samples",
      lineChart([{ pts: cp, c: "var(--e-acc)", w: 1.5, dots: true, dotScale: Math.max(1, Math.max(...bins.map(b => b.n)) / 50), title: p => `predicted ${pct(p[0], 1)} · realised ${pct(p[1], 1)} · n=${p[2]}` }], 1, { diag: true }).replace(/>(\d+)%<\/text>/g, ">$1%</text>")
      + `<div class="note">x axis: predicted (0–1), y axis: realised. ${cp.reduce((a, p) => a + p[2], 0)} samples.</div>`));
    // searches
    let full = 0, cheap = 0; const vc = {};
    for (const s of S) if (s) for (const v of s.visits) { vc[v] = (vc[v] || 0) + 1; if (fullVisits !== null && v < fullVisits) cheap++; else full++; }
    const vk = Object.keys(vc).map(Number).sort((a, b) => a - b), tot = full + cheap;
    const vtable = vk.length <= 14 ? `<table class="st"><tr><th>visits</th><th>searches</th><th>share</th></tr>${vk.map(v => `<tr><td>${v}</td><td>${vc[v]}</td><td>${pct(vc[v] / tot, 1)}</td></tr>`).join("")}</table>` : histogram(vk.flatMap(v => Array(vc[v]).fill(v)), Math.max(1, Math.ceil(vk[vk.length - 1] / 20)), vk[vk.length - 1], "var(--link)");
    cards.push(card("Searches", `Full searches (visits ≥ ${fullVisits}, the median of the weighted ones) vs cheaper ones: ${full} full, ${cheap} cheaper (${pct(cheap / (tot || 1), 1)})`, vtable));
    const decided = sEval.filter(s => s.minWinnerProb !== null), comeback = decided.filter(s => s.minWinnerProb < 0.1).length;
    cards.push(card("Comebacks", "Decided games where the winner's win probability was below 10% at some ply of the search", `<div class="mini"><b>${comeback}</b> of ${decided.length} decided games with search values (${pct(comeback / (decided.length || 1), 1)})</div>`));
  } else {
    cards.push(card("Search values", "", "<div class=note>No move of these games carries a search comment.</div>"));
  }
  $("statCards").innerHTML = cards.join("");
  renderMatchStats(idx, H);
}

function renderMatchStats(idx, H) {
  const sec = $("matchSection");
  const mg = H.filter(h => h.table);
  if (!mg.length) { sec.hidden = true; return; }
  sec.hidden = false;
  $("matchTitle").innerHTML = `Match results <small>${mg.length} games with a table; score = points per seat: win 1, draw ¼, loss 0 (25% = even); interval: Wilson 95%</small>`;
  const P = {};
  const player = (name, s, h) => { const k = name; return P[k] = P[k] || { name: k, models: new Set(), seats: 0, pts: 0, wins: 0, bySeat: [0, 0, 0, 0], seatN: [0, 0, 0, 0] }; };
  for (const h of mg) {
    if (!h.finished) continue;
    for (let s = 0; s < 4; s++) {
      const p = player(h.names[s], s, h), pts = seatPoints(h, s);
      if (modelOf(h, s)) p.models.add(modelOf(h, s));
      p.seats++; p.pts += pts; p.seatN[s]++;
      if (h.winner === s) { p.wins++; p.bySeat[s]++; }
    }
  }
  const rows = Object.values(P).sort((a, b) => b.pts / b.seats - a.pts / a.seats).map(p => {
    const [lo, hi] = wilson(p.pts, p.seats);
    return `<tr><td>${escapeHtml(p.name)}${p.models.size ? `<br><small>${escapeHtml([...p.models].join(", "))}</small>` : ""}</td><td>${p.seats}</td><td>${p.wins}</td><td><b>${pct(p.pts / p.seats, 1)}</b></td><td>${pct(lo, 1)}–${pct(hi, 1)}</td>${p.bySeat.map((w, s) => `<td>${w}/${p.seatN[s]}</td>`).join("")}</tr>`;
  });
  const T = {};
  for (const h of mg) {
    const t = T[h.table] = T[h.table] || { games: 0, plies: 0, seatWins: [0, 0, 0, 0], draws: {}, rot: {} };
    t.games++; t.plies += h.plies;
    const r = t.rot[h.rotation] = t.rot[h.rotation] || { games: 0, seatWins: [0, 0, 0, 0], draws: 0 };
    r.games++;
    if (h.winner >= 0) { t.seatWins[h.winner]++; r.seatWins[h.winner]++; }
    else { const k = h.finished ? h.drawReason : "unfinished"; t.draws[k] = (t.draws[k] || 0) + 1; r.draws++; }
  }
  const trow = Object.keys(T).sort().map(k => {
    const t = T[k], d = Object.entries(t.draws).map(([a, b]) => `${a} ${b}`).join(", ") || "0";
    return `<tr><td>${escapeHtml(k)}</td><td>${t.games}</td>${t.seatWins.map(x => `<td>${x}</td>`).join("")}<td>${d}</td><td>${(t.plies / t.games).toFixed(1)}</td></tr>`;
  });
  const rrow = [];
  for (const k of Object.keys(T).sort()) for (const r of Object.keys(T[k].rot).sort()) { const x = T[k].rot[r]; rrow.push(`<tr><td>${escapeHtml(k)}</td><td>${r}</td><td>${x.games}</td>${x.seatWins.map(w => `<td>${w}</td>`).join("")}<td>${x.draws}</td></tr>`); }
  $("matchCards").innerHTML =
    card("Players", "Per player name over its seats in the finished games (wins by seat: wins / appearances in that seat)",
      `<table class="st"><tr><th>player</th><th>seats</th><th>wins</th><th>score</th><th>95% interval</th><th>S</th><th>W</th><th>N</th><th>E</th></tr>${rows.join("")}</table>`, true)
    + card("Tables", "Results per table (like match_report.py)", `<table class="st"><tr><th>table</th><th>games</th><th>S wins</th><th>W wins</th><th>N wins</th><th>E wins</th><th>draws</th><th>mean plies</th></tr>${trow.join("")}</table>`, true)
    + card("Seat rotations", "Seat wins per table and rotation (rotation r puts slot (s + r) mod 4 in seat s)", `<table class="st"><tr><th>table</th><th>rotation</th><th>games</th><th>S</th><th>W</th><th>N</th><th>E</th><th>draws</th></tr>${rrow.join("")}</table>`);
}
