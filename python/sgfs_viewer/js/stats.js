"use strict";
// ---- Statistics ------------------------------------------------------------
function summarize(line, h) {
  const g = parseGame(line);
  const s = {
    i: h.i, n: g.moves.length, start: g.startTurnIdx, searched: g.moves.length - g.startTurnIdx,
    winner: h.winner, gtype: h.gtype, margin: Math.abs(parseFloat(h.result.slice(2)) || 0),
    used: { B: 0, W: 0 }, hw: 0, vw: 0, pawnMoves: 0, evs: [], firstWallSearch: null,
    firstToMove: g.moves[g.startTurnIdx]?.pla ?? null,
    pb: h.pb, pw: h.pw, winnerModel: h.winnerModel, komi: h.komi, wb: h.wb, ww: h.ww, rulesLabel: h.rulesLabel,
  };
  s.signed = s.winner === "W" ? s.margin : -s.margin;
  s.walls = 0;
  // path race: shortest-path length to goal for both pawns after every move (no jumps)
  const bl = newBlock(), pos = { B: { ...g.startB }, W: { ...g.startW } };
  const dG = { B: goalDist(bl, "B"), W: goalDist(bl, "W") };
  const loser = s.winner === "B" ? "W" : "B";
  s.raceDeficit = -Infinity; s.maxPath = 0;
  const race = () => {
    const d = { B: dG.B[cellOf(pos.B)], W: dG.W[cellOf(pos.W)] };
    if (s.winner === "B" || s.winner === "W") s.raceDeficit = Math.max(s.raceDeficit, d[s.winner] - d[loser]);
    s.maxPath = Math.max(s.maxPath, d.B, d.W);
  };
  race();
  g.moves.forEach((mv, i) => {
    const k = moveKind(mv);
    if (k === "pawn") { s.pawnMoves++; pos[mv.pla] = { x: mv.x, y: mv.y }; }
    else if (k === "hwall" || k === "vwall") {
      s.used[mv.pla]++; s.walls++; k === "hwall" ? s.hw++ : s.vw++;
      if (i >= g.startTurnIdx && s.firstWallSearch === null) s.firstWallSearch = i + 1;
      addWallToBlock(bl, k, mv.x, mv.y);
      for (const p of ["B", "W"]) if (wallAffects(dG[p], k, mv.x, mv.y)) dG[p] = goalDist(bl, p);
    }
    if (hasEv(mv)) s.evs.push({ i, pla: mv.pla, p: mv.ev.wWin, score: mv.ev.wScore, v: mv.ev.v, w: mv.ev.weight });
    race();
  });
  if (!isFinite(s.raceDeficit)) s.raceDeficit = null;
  // eval-based metrics, all over the searched positions, from the eventual winner's view where it matters
  const E = s.evs, has = E.length > 0;
  const pw = e => s.winner === "W" ? e.p : 1 - e.p;
  const sw = e => s.winner === "W" ? e.score : -e.score;
  s.minWinnerP = has ? Math.min(...E.map(pw)) : null;
  s.minWinnerScore = has ? Math.min(...E.map(sw)) : null;
  let lastBelow = -1, lead = 0, vol = 0, swing = 0, brier = 0, prevSide = 0;
  E.forEach((e, k) => {
    if (pw(e) < 0.9) lastBelow = e.i;
    const side = e.p > 0.5 ? 1 : e.p < 0.5 ? -1 : 0;
    if (side && prevSide && side !== prevSide) lead++;
    if (side) prevSide = side;
    if (k) { const d = Math.abs(e.p - E[k - 1].p); vol += d; swing = Math.max(swing, d); }
    brier += (e.p - (s.winner === "W" ? 1 : 0)) ** 2;
  });
  s.decidedLeft = has ? s.n - (lastBelow + 1) : null;  // moves remaining once decided
  s.leadChanges = has ? lead : null;
  s.volatility = has ? vol : null;
  s.maxSwing = has ? swing : null;
  s.brier = has ? brier / E.length : null;
  return s;
}

function quant(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function describe(arr) {
  const a = [...arr].sort((x, y) => x - y), n = a.length;
  const mean = a.reduce((s, x) => s + x, 0) / (n || 1);
  const sd = Math.sqrt(a.reduce((s, x) => s + (x - mean) ** 2, 0) / (n || 1));
  return { n, mean, sd, median: quant(a, .5), min: a[0], max: a[n - 1], p10: quant(a, .1), p25: quant(a, .25), p75: quant(a, .75), p90: quant(a, .9) };
}
function niceWidth(range, target = 36) {
  const raw = Math.max(1, range / target);
  for (const s of [1, 2, 5, 10, 20, 25, 50, 100]) if (s >= raw) return s;
  return 200;
}
// integer-valued histogram with shared bins. Values that are all half-integers (SGF results since the Quoridor I/O v2
// rules: the lead, e.g. W+0.5 or B+2.5) get bins shifted by 0.5, so each bin is labelled by the value it holds.
function binsFor(arrays, width, lo, hi) {
  const all = arrays.flat();
  const half = all.length > 0 && all.every(x => Math.abs(x - Math.floor(x) - 0.5) < 1e-9) ? 0.5 : 0;
  lo ??= Math.floor((Math.min(...all) - half) / width) * width + half;
  hi ??= Math.max(...all);
  const nb = Math.floor((hi - lo) / width) + 1;
  const counts = arrays.map(a => { const c = new Array(nb).fill(0); for (const x of a) c[Math.min(nb - 1, Math.floor((x - lo) / width))]++; return c; });
  const labels = Array.from({ length: nb }, (_, k) => width === 1 ? `${lo + k}` : `${lo + k * width}–${lo + (k + 1) * width - 1}`);
  return { lo, width, nb, counts, labels };
}

// Bar chart. series: [{name, color, values}] drawn grouped per category.
// opts: labels, height, yMax, yFmt, markers [{at (category units, fractional), label}], tickEvery, tipFor(k), colorsFor(k, s)
function barChart(host, series, opts) {
  const W = 560, H = opts.height || 170, L = 40, R = 10, T = 16, B = 26;
  const nb = opts.labels.length, pw = W - L - R, ph = H - T - B, band = pw / nb;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, host);
  let yMax = opts.yMax ?? Math.max(1e-9, ...series.flatMap(s => s.values));
  const Y = v => T + ph - (v / yMax) * ph;
  const yFmt = opts.yFmt || (v => Math.round(v));
  for (let k = 0; k <= 4; k++) {
    const v = yMax * k / 4, y = Y(v);
    el("line", { x1: L, x2: W - R, y1: y, y2: y, stroke: k ? "var(--grid)" : "#d6d1c5" }, svg);
    txt(svg, L - 5, y + 3.5, yFmt(v), { "text-anchor": "end" });
  }
  const k = series.length, gw = Math.max(1, band - 2);
  const bw = Math.max(1, (gw - (k - 1) * 2) / k);
  for (let b = 0; b < nb; b++) {
    series.forEach((s, j) => {
      const v = s.values[b]; if (!v) return;
      const x = L + b * band + 1 + j * (bw + 2);
      el("path", { d: barPath(x, Y(v), bw, T + ph - Y(v), Math.min(4, bw / 2)), fill: opts.colorsFor ? opts.colorsFor(b, s) : s.color }, svg);
    });
  }
  const every = opts.tickEvery || Math.max(1, Math.ceil(nb / 10));
  for (let b = 0; b < nb; b += every) txt(svg, L + (b + 0.5) * band, H - 9, opts.tickLabels ? opts.tickLabels[b] : opts.labels[b].split("–")[0], { "text-anchor": "middle" });
  for (const m of opts.markers || []) {
    const x = L + m.at * band;
    el("line", { x1: x, x2: x, y1: T - 4, y2: T + ph, stroke: "#3e4c59", "stroke-width": 1.2, "stroke-dasharray": "4 3" }, svg);
    txt(svg, x + (m.anchor === "end" ? -4 : 4), T + 4, m.label, { fill: "#3e4c59", "font-weight": 600, "text-anchor": m.anchor || "start" });
  }
  // hover: full-height hit target per category
  for (let b = 0; b < nb; b++) {
    const hit = el("rect", { x: L + b * band, y: T, width: band, height: ph, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", e => showTip(e, opts.tipFor ? opts.tipFor(b) : `<b>${opts.labels[b]}</b><br>` + series.map(s => `${sw(s.color)}${s.name}: ${yFmt(s.values[b])}`).join("<br>")));
    hit.addEventListener("mouseleave", hideTip);
  }
  return svg;
}

function card(title, desc, parent = $("statCards")) {
  const c = document.createElement("div");
  c.className = "card";
  c.innerHTML = `<h3>${title}</h3><div class="desc">${desc}</div>`;
  parent.appendChild(c);
  return c;
}
function legend(host, items) {
  const d = document.createElement("div");
  d.className = "legend dots";
  d.innerHTML = items.map(([n, c]) => `<span style="--c:${c}">${n}</span>`).join("");
  host.appendChild(d);
}
function mini(host, html) { const d = document.createElement("div"); d.className = "mini"; d.innerHTML = html; host.appendChild(d); }
const f1 = x => isFinite(x) ? x.toFixed(1) : "—";
const markersFor = (d, bins) => [
  { at: (d.median - bins.lo) / bins.width + (bins.width === 1 ? 0.5 : 0), label: `median ${f1(d.median)}` },
  { at: (d.mean - bins.lo) / bins.width + (bins.width === 1 ? 0.5 : 0), label: `mean ${f1(d.mean)}`, anchor: "end" },
].sort((a, b) => a.at - b.at).map((m, k) => ({ ...m, anchor: k ? "start" : "end" }));

function renderStats() {
  if (!games.length) { $("statsNote").textContent = "Load a file first."; return; }
  if (!statsCache) { $("statsNote").textContent = "Computing game metrics…"; return; }   // computeSummaries() re-renders when done
  const t = $("sType").value;
  const S = statsCache.filter(s => !t || s.gtype === t);
  $("statTiles").innerHTML = ""; $("statCards").innerHTML = ""; $("matchTiles").innerHTML = ""; $("matchCards").innerHTML = "";
  $("statsNote").textContent = `${S.length} games`;
  $("matchSection").hidden = $("genTitle").hidden = !match;
  if (!S.length) return;
  if (match) renderMatchStats(S);
  const B = S.filter(s => s.winner === "B"), Wn = S.filter(s => s.winner === "W");
  const C_B = "var(--b-acc)", C_W = "var(--w-acc)";
  const dAll = describe(S.map(s => s.n)), dB = describe(B.map(s => s.n)), dW = describe(Wn.map(s => s.n));
  const dS = describe(S.map(s => s.searched));

  // calibration
  const calib = Array.from({ length: 10 }, () => ({ n: 0, sumP: 0, wins: 0 }));
  let brier = 0, nPos = 0, fullV = 0, maxV = 0, sumW = 0, nW = 0;
  for (const s of S) for (const e of s.evs) if (e.v > maxV) maxV = e.v;
  for (const s of S) for (const e of s.evs) {
    const y = s.winner === "W" ? 1 : 0, b = Math.min(9, Math.floor(e.p * 10));
    calib[b].n++; calib[b].sumP += e.p; calib[b].wins += y;
    brier += (e.p - y) ** 2; nPos++;
    if (e.v === maxV) fullV++;
    if (e.w !== undefined) { sumW += e.w; nW++; }
  }
  const comeback = S.filter(s => s.minWinnerP !== null && s.minWinnerP < 0.2).length;
  const withEv = S.filter(s => s.minWinnerP !== null).length;

  const tiles = [
    ["Games", S.length, `${B.length} B · ${Wn.length} W`],
    ["Black (1st) wins", pct(B.length / S.length), `White (2nd) ${pct(Wn.length / S.length)}`],
    ["Game length", f1(dAll.mean), `± ${f1(dAll.sd)} · median ${f1(dAll.median)}`],
    ["Length range", `${dAll.min}–${dAll.max}`, `p10 ${f1(dAll.p10)} · p90 ${f1(dAll.p90)}`],
    ["Searched moves", f1(dS.mean), `± ${f1(dS.sd)} · median ${f1(dS.median)}`],
    ["Final margin", f1(describe(S.map(s => s.margin)).mean), `median ${f1(describe(S.map(s => s.margin)).median)}`],
    ["Walls used", `${f1(describe(S.map(s => s.used.B)).mean)} / ${f1(describe(S.map(s => s.used.W)).mean)}`, "mean per game, B / W"],
    ["Brier score", nPos ? (brier / nPos).toFixed(3) : "—", `${nPos} searched positions`],
    ["Comebacks", withEv ? pct(comeback / withEv) : "—", "winner once < 20% win"],
    ["Full searches", nPos ? pct(fullV / nPos) : "—", `v=${maxV} · mean weight ${nW ? (sumW / nW).toFixed(2) : "—"}`],
  ];
  $("statTiles").innerHTML = tiles.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  // 1. game length histogram
  {
    const c = card("Game length", "Total moves per game (including random init). Dashed lines: median and mean.");
    const w = niceWidth(dAll.max - dAll.min);
    const bins = binsFor([S.map(s => s.n)], w);
    mini(c, `mean ${f1(dAll.mean)} · median ${f1(dAll.median)} · sd ${f1(dAll.sd)} · IQR ${f1(dAll.p25)}–${f1(dAll.p75)}`);
    barChart(c, [{ name: "Games", color: "var(--neutral)", values: bins.counts[0] }], { labels: bins.labels, markers: markersFor(dAll, bins),
      tipFor: b => `<b>${bins.labels[b]} moves</b><br>${bins.counts[0][b]} games (${pct(bins.counts[0][b] / S.length)})` });
  }

  // 2. game length by winner, small multiples on shared scales
  {
    const c = card("Game length by winner", "Same bins and y-scale for both panels, so the shapes compare directly.");
    const lo = Math.min(dB.min ?? Infinity, dW.min ?? Infinity), hi = Math.max(dB.max ?? 0, dW.max ?? 0);
    const w = niceWidth(hi - lo);
    const bins = binsFor([B.map(s => s.n), Wn.map(s => s.n)], w, Math.floor(lo / w) * w, hi);
    const yMax = Math.max(...bins.counts[0], ...bins.counts[1], 1);
    [["Black (1st) wins", C_B, 0, dB, B.length], ["White (2nd) wins", C_W, 1, dW, Wn.length]].forEach(([name, col, j, d, n]) => {
      legend(c, [[`${name} · ${n} games`, col]]);
      mini(c, n ? `mean ${f1(d.mean)} · median ${f1(d.median)} · sd ${f1(d.sd)} · range ${d.min}–${d.max}` : "none");
      if (n) barChart(c, [{ name, color: col, values: bins.counts[j] }], { labels: bins.labels, yMax, height: 130, markers: markersFor(d, bins),
        tipFor: b => `<b>${bins.labels[b]} moves</b><br>${sw(col)}${bins.counts[j][b]} games (${pct(bins.counts[j][b] / n)} of ${name.toLowerCase()})` });
    });
  }

  // 3. win rates by slice
  {
    const c = card("Win rate: Black (1st) vs White (2nd)", "Share of games won by Black; the rest is White's wins and draws. Overall and by slice, including by komi and initial walls when some games are not standard.");
    legend(c, [["Black (1st)", C_B], ["White (2nd)", C_W]]);
    const rows = [["All games", S]];
    for (const gt of [...new Set(S.map(s => s.gtype))].sort()) rows.push([`gtype = ${gt || "—"}`, S.filter(s => s.gtype === gt)]);
    rows.push(["Black moves first after init", S.filter(s => s.firstToMove === "B")]);
    rows.push(["White moves first after init", S.filter(s => s.firstToMove === "W")]);
    const q = [dAll.p25, dAll.median, dAll.p75];
    rows.push([`Short games (≤ ${f1(q[0])})`, S.filter(s => s.n <= q[0])]);
    rows.push([`Long games (> ${f1(q[2])})`, S.filter(s => s.n > q[2])]);
    // Quoridor I/O v2 self-play and arena games can have a komi and a fence handicap: slices by komi (10/10 walls)
    // and by initial walls. Positive komi favours White. Shown only when some games are non-standard.
    if (S.some(s => s.rulesLabel)) {
      const tenTen = S.filter(s => s.wb === STANDARD_WALLS && s.ww === STANDARD_WALLS);
      for (const k of [...new Set(tenTen.map(s => s.komi))].sort((a, b) => a - b))
        rows.push([`Komi ${fmtKomi(k)}${k === STANDARD_KOMI ? " (standard)" : ""}, walls 10/10`, tenTen.filter(s => s.komi === k)]);
      const hc = S.filter(s => s.wb !== STANDARD_WALLS || s.ww !== STANDARD_WALLS);
      rows.push(["Black has fewer walls", hc.filter(s => s.wb < s.ww)]);
      rows.push(["White has fewer walls", hc.filter(s => s.ww < s.wb)]);
    }
    for (const [lab, arr] of rows) {
      if (!arr.length) continue;
      const b = arr.filter(s => s.winner === "B").length / arr.length;
      const nDraw = arr.filter(s => s.winner !== "B" && s.winner !== "W").length;
      const d = document.createElement("div");
      d.className = "wr-row";
      d.innerHTML = `<div class="lab">${lab}<small>${arr.length} games${nDraw ? ` · ${nDraw} draws` : ""}</small></div><div><div class="wr-bar"><div style="flex-basis:${100 * b}%;background:${C_B}"></div><div style="flex:1;background:${C_W}"></div></div><div class="wr-nums"><span>B ${pct(b)}</span><span>W ${pct(1 - b)}</span></div></div>`;
      c.appendChild(d);
    }
  }

  // 4. black win% by length bin
  {
    const c = card("Black (1st) win% vs game length", "Per length bin; bins with fewer than 10 games are left out. Dashed line: 50%.");
    const w = niceWidth(dAll.max - dAll.min, 20);
    const bins = binsFor([S.map(s => s.n), B.map(s => s.n)], w, Math.floor(dAll.min / w) * w, dAll.max);
    const rate = bins.counts[0].map((n, k) => n >= 10 ? bins.counts[1][k] / n : 0);
    const svg = barChart(c, [{ name: "Black win%", color: C_B, values: rate }], { labels: bins.labels, yMax: 1, yFmt: v => Math.round(v * 100) + "%",
      tipFor: k => `<b>${bins.labels[k]} moves</b><br>${bins.counts[0][k]} games` + (bins.counts[0][k] >= 10 ? `<br>${sw(C_B)}Black wins ${pct(rate[k])}` : "<br>(too few games)") });
    const y = 16 + (170 - 16 - 26) * 0.5;
    el("line", { x1: 40, x2: 550, y1: y, y2: y, stroke: "#3e4c59", "stroke-dasharray": "4 3", "stroke-width": 1 }, svg);
  }

  // 5. signed final margin
  {
    const c = card("Final margin (White's view)", "Result margin (the SGF result: since the I/O v2 rules the tempo lead, e.g. W+0.5); negative = Black won by that much, positive = White won. Draws are left out.");
    legend(c, [["Black wins", C_B], ["White wins", C_W]]);
    const decided = [...B, ...Wn], draws = S.length - decided.length;
    const vals = decided.map(s => s.signed), lo = Math.min(...vals), hi = Math.max(...vals);
    const w = niceWidth(hi - lo, 40);
    const bins = binsFor([vals], w);
    const d = describe(vals);
    mini(c, `mean ${fmtScore(d.mean)} · median ${fmtScore(d.median)} · mean |margin| B wins ${f1(describe(B.map(s => s.margin)).mean)}, W wins ${f1(describe(Wn.map(s => s.margin)).mean)}` + (draws ? ` · ${draws} draws` : ""));
    barChart(c, [{ name: "Games", color: C_W, values: bins.counts[0] }], { labels: bins.labels,
      colorsFor: b => bins.lo + b * w < 0 ? C_B : C_W,
      tipFor: b => `<b>margin ${bins.labels[b]}</b><br>${bins.counts[0][b]} games` });
  }

  // 6. walls used by winner vs loser
  {
    const c = card("Walls used: winner vs loser", "Walls placed per game (0–10) by the winning and the losing side. Draws are left out.");
    legend(c, [["Winner", "var(--margin)"], ["Loser", "var(--neutral)"]]);
    const decided = [...B, ...Wn];
    const win = decided.map(s => s.used[s.winner]), lose = decided.map(s => s.used[s.winner === "B" ? "W" : "B"]);
    const bins = binsFor([win, lose], 1, 0, WALLS_PER_PLAYER);
    const hv = S.reduce((a, s) => [a[0] + s.hw, a[1] + s.vw], [0, 0]);
    mini(c, `mean winner ${f1(describe(win).mean)} · loser ${f1(describe(lose).mean)} · horizontal ${pct(hv[0] / (hv[0] + hv[1] || 1))} of walls`);
    barChart(c, [{ name: "Winner", color: "var(--margin)", values: bins.counts[0] }, { name: "Loser", color: "var(--neutral)", values: bins.counts[1] }], { labels: bins.labels, tickEvery: 1 });
  }

  // 7. walls left at the end, by side
  {
    const c = card("Walls used by side", "Walls placed per game by Black (1st) and White (2nd).");
    legend(c, [["Black (1st)", C_B], ["White (2nd)", C_W]]);
    const bins = binsFor([S.map(s => s.used.B), S.map(s => s.used.W)], 1, 0, WALLS_PER_PLAYER);
    barChart(c, [{ name: "Black", color: C_B, values: bins.counts[0] }, { name: "White", color: C_W, values: bins.counts[1] }], { labels: bins.labels, tickEvery: 1 });
  }

  // 8. calibration
  if (nPos) {
    const c = card("Win% calibration", "Searched positions binned by predicted White win%; bar = actual White win rate in that bin. Dashed = perfect calibration.");
    const labels = calib.map((_, k) => `${k * 10}–${k * 10 + 10}%`);
    const actual = calib.map(b => b.n ? b.wins / b.n : 0);
    const svg = barChart(c, [{ name: "Actual White win%", color: C_W, values: actual }], { labels, yMax: 1, yFmt: v => Math.round(v * 100) + "%", tickEvery: 1,
      tickLabels: labels.map(l => l.split("–")[0]),
      tipFor: k => `<b>predicted ${labels[k]}</b><br>${calib[k].n} positions<br>mean predicted ${calib[k].n ? pct(calib[k].sumP / calib[k].n) : "—"}<br>${sw(C_W)}actual ${calib[k].n ? pct(actual[k]) : "—"}` });
    // diagonal through bin centers (predicted mean)
    const L = 40, band = (560 - 40 - 10) / 10, T = 16, ph = 170 - 16 - 26;
    el("line", { x1: L, y1: T + ph, x2: L + 10 * band, y2: T, stroke: "#3e4c59", "stroke-dasharray": "4 3", "stroke-width": 1.2, "pointer-events": "none" }, svg);
  }

  // 9. how early is the game decided
  {
    const c = card("Moves left after the game is decided", "Moves from the last search where the eventual winner was below 90% until the end.");
    const vals = S.filter(s => s.decidedLeft !== null).map(s => s.decidedLeft);
    if (vals.length) {
      const d = describe(vals), w = niceWidth(d.max - d.min);
      const bins = binsFor([vals], w, 0, d.max);
      mini(c, `mean ${f1(d.mean)} · median ${f1(d.median)} · sd ${f1(d.sd)}`);
      barChart(c, [{ name: "Games", color: "var(--margin)", values: bins.counts[0] }], { labels: bins.labels, markers: markersFor(d, bins),
        tipFor: b => `<b>${bins.labels[b]} moves left</b><br>${bins.counts[0][b]} games` });
    }
  }

  // 10. random init length + first searched wall
  {
    const c = card("Random init length", "startTurnIdx: moves played randomly before search took over.");
    const vals = S.map(s => s.start), d = describe(vals);
    const bins = binsFor([vals], niceWidth(d.max - d.min), 0, d.max);
    mini(c, `mean ${f1(d.mean)} · median ${f1(d.median)} · range ${d.min}–${d.max}`);
    barChart(c, [{ name: "Games", color: "var(--neutral)", values: bins.counts[0] }], { labels: bins.labels,
      tipFor: b => `<b>init ${bins.labels[b]}</b><br>${bins.counts[0][b]} games` });
  }
  {
    const c = card("First wall during search", "Move number of the first wall placed after random init (games with no searched wall excluded).");
    const vals = S.filter(s => s.firstWallSearch !== null).map(s => s.firstWallSearch);
    if (vals.length) {
      const d = describe(vals), bins = binsFor([vals], niceWidth(d.max - d.min));
      mini(c, `mean ${f1(d.mean)} · median ${f1(d.median)} · ${pct(1 - vals.length / S.length)} of games never place a wall during search`);
      barChart(c, [{ name: "Games", color: "var(--neutral)", values: bins.counts[0] }], { labels: bins.labels, markers: markersFor(d, bins),
        tipFor: b => `<b>move ${bins.labels[b]}</b><br>${bins.counts[0][b]} games` });
    }
  }

  // 11. summary table
  {
    const c = card("Length summary table", "Total moves per game.");
    const row = (lab, d) => `<tr><td>${lab}</td><td>${d.n}</td><td>${f1(d.mean)}</td><td>${f1(d.sd)}</td><td>${d.min ?? "—"}</td><td>${f1(d.p25)}</td><td>${f1(d.median)}</td><td>${f1(d.p75)}</td><td>${d.max ?? "—"}</td></tr>`;
    c.insertAdjacentHTML("beforeend", `<table class="st"><tr><th></th><th>n</th><th>mean</th><th>sd</th><th>min</th><th>p25</th><th>median</th><th>p75</th><th>max</th></tr>` +
      row("All", dAll) + row("Black wins", dB) + row("White wins", dW) + row("Searched only", dS) + `</table>`);
  }
}

// ---- Match (two different models) statistics -------------------------------
function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
  const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}
const elo = p => p <= 0 ? -Infinity : p >= 1 ? Infinity : -400 * Math.log10(1 / p - 1);
const fmtElo = e => !isFinite(e) ? (e > 0 ? "+∞" : "−∞") : (e >= 0 ? "+" : "−") + Math.abs(e).toFixed(0);
function erfc(x) {  // Abramowitz–Stegun 7.1.26
  const t = 1 / (1 + 0.3275911 * x);
  return t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-x * x);
}
// two-sided binomial p-value vs 50% (normal approximation with continuity correction)
const pTwoSided = (k, n) => n ? Math.min(1, erfc(Math.max(0, Math.abs(k - n / 2) - 0.5) / Math.sqrt(n / 4) / Math.SQRT2)) : 1;

function wrRow(host, lab, sub, p, cA, cB, nameA, nameB) {
  const d = document.createElement("div");
  d.className = "wr-row";
  d.innerHTML = `<div class="lab">${lab}<small>${sub}</small></div><div><div class="wr-bar"><div style="flex-basis:${100 * p}%;background:${cA}"></div><div style="flex:1;background:${cB}"></div></div><div class="wr-nums"><span>${nameA} ${pct(p)}</span><span>${nameB} ${pct(1 - p)}</span></div></div>`;
  host.appendChild(d);
}

function renderMatchStats(S) {
  const [A, Bm] = match.models, [LA, LB] = match.labels, [CA, CB] = match.colors;
  const host = $("matchCards");
  $("matchTitle").innerHTML = `Match · ${escapeHtml(LA)} vs ${escapeHtml(LB)} <small>${escapeHtml(LA)} = ${escapeHtml(A)} · ${escapeHtml(LB)} = ${escapeHtml(Bm)}` +
    (match.labels[0] === "New" ? " · newer = more training samples (-s)" : "") + `</small>`;
  const winsOf = (arr, m) => arr.filter(s => s.winnerModel === m).length;
  const aB = S.filter(s => s.pb === A), aW = S.filter(s => s.pw === A);
  const kA = winsOf(S, A), n = S.length, pA = kA / n, [lo, hi] = wilson(kA, n);
  const kAB = winsOf(aB, A), kAW = winsOf(aW, A);
  const blackWins = S.filter(s => s.winner === "B").length;
  const pv = pTwoSided(kA, n);
  const tiles = [
    ["Games", n, `${LA} as Black ${aB.length} · as White ${aW.length}`],
    [`${LA} score`, `${kA}–${n - kA}`, `${pct(pA)} · 95% CI ${pct(lo)}–${pct(hi)}`],
    [`Elo ${LA} − ${LB}`, fmtElo(elo(pA)), `95% CI ${fmtElo(elo(lo))} … ${fmtElo(elo(hi))}`],
    ["p vs 50%", pv < 0.001 ? "<0.001" : pv.toFixed(3), pv < 0.05 ? "significant at 5%" : "not significant at 5%"],
    [`${LA} as Black (1st)`, `${kAB}/${aB.length}`, aB.length ? pct(kAB / aB.length) : "—"],
    [`${LA} as White (2nd)`, `${kAW}/${aW.length}`, aW.length ? pct(kAW / aW.length) : "—"],
    ["Black (1st) wins", pct(blackWins / n), "color bias in this match"],
  ];
  $("matchTiles").innerHTML = tiles.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  // head-to-head
  {
    const c = card("Head-to-head", `Share of games won by ${LA} and ${LB}, overall and by slice.`, host);
    legend(c, [[LA, CA], [LB, CB]]);
    const d = describe(S.map(s => s.n));
    const rows = [["All games", S], [`${LA} as Black (1st)`, aB], [`${LA} as White (2nd)`, aW],
      [`Short games (≤ ${f1(d.p25)})`, S.filter(s => s.n <= d.p25)], [`Long games (> ${f1(d.p75)})`, S.filter(s => s.n > d.p75)]];
    for (const gt of [...new Set(S.map(s => s.gtype))].sort()) if (new Set(S.map(s => s.gtype)).size > 1) rows.push([`gtype = ${gt}`, S.filter(s => s.gtype === gt)]);
    for (const [lab, arr] of rows) if (arr.length) {
      const k = winsOf(arr, A), [l, h] = wilson(k, arr.length);
      wrRow(c, lab, `${arr.length} games · CI ${pct(l, 0)}–${pct(h, 0)}`, k / arr.length, CA, CB, LA, LB);
    }
  }

  // record table
  {
    const c = card("Record by color", `Rows are from ${LA}'s side. Margin = mean result margin of the games won.`, host);
    const row = (lab, arr) => {
      const won = arr.filter(s => s.winnerModel === A), lost = arr.filter(s => s.winnerModel !== A);
      const mm = a => a.length ? f1(describe(a.map(s => s.margin)).mean) : "—";
      return `<tr><td>${lab}</td><td>${arr.length}</td><td>${won.length}</td><td>${lost.length}</td><td>${arr.length ? pct(won.length / arr.length) : "—"}</td><td>${arr.length ? fmtElo(elo(won.length / arr.length)) : "—"}</td><td>${arr.length ? f1(describe(arr.map(s => s.n)).mean) : "—"}</td><td>${mm(won)}</td><td>${mm(lost)}</td></tr>`;
    };
    c.insertAdjacentHTML("beforeend", `<table class="st"><tr><th></th><th>games</th><th>${LA} W</th><th>${LA} L</th><th>win%</th><th>Elo</th><th>mean len</th><th>margin (won)</th><th>margin (lost)</th></tr>` +
      row(`${LA} as Black`, aB) + row(`${LA} as White`, aW) + row("Total", S) + `</table>`);
  }

  // length by winning model
  {
    const c = card("Game length by winning model", "Same bins and y-scale for both panels.", host);
    const WA = S.filter(s => s.winnerModel === A), WB = S.filter(s => s.winnerModel === Bm);
    const all = S.map(s => s.n), lo = Math.min(...all), hi = Math.max(...all), w = niceWidth(hi - lo, 30);
    const bins = binsFor([WA.map(s => s.n), WB.map(s => s.n)], w, Math.floor(lo / w) * w, hi);
    const yMax = Math.max(1, ...bins.counts[0], ...bins.counts[1]);
    [[`${LA} wins`, CA, 0, WA], [`${LB} wins`, CB, 1, WB]].forEach(([name, col, j, arr]) => {
      legend(c, [[`${name} · ${arr.length} games`, col]]);
      if (!arr.length) return mini(c, "none");
      const d = describe(arr.map(s => s.n));
      mini(c, `mean ${f1(d.mean)} · median ${f1(d.median)} · sd ${f1(d.sd)} · range ${d.min}–${d.max}`);
      barChart(c, [{ name, color: col, values: bins.counts[j] }], { labels: bins.labels, yMax, height: 120, markers: markersFor(d, bins),
        tipFor: b => `<b>${bins.labels[b]} moves</b><br>${sw(col)}${bins.counts[j][b]} games` });
    });
  }

  // signed margin from A's view
  {
    const c = card(`Final margin (${LA}'s view)`, `Negative = ${LB} won by that much, positive = ${LA} won.`, host);
    legend(c, [[`${LB} wins`, CB], [`${LA} wins`, CA]]);
    const decided = S.filter(s => s.winnerModel), draws = S.length - decided.length;
    const vals = decided.map(s => s.winnerModel === A ? s.margin : -s.margin), lo = Math.min(...vals), hi = Math.max(...vals);
    const w = niceWidth(hi - lo, 40), bins = binsFor([vals], w), d = describe(vals);
    mini(c, `mean ${fmtScore(d.mean)} · median ${fmtScore(d.median)} · sd ${f1(d.sd)}` + (draws ? ` · ${draws} draws (left out)` : ""));
    barChart(c, [{ name: "Games", color: CA, values: bins.counts[0] }], { labels: bins.labels,
      colorsFor: b => bins.lo + b * w < 0 ? CB : CA, tipFor: b => `<b>margin ${bins.labels[b]}</b><br>${bins.counts[0][b]} games` });
  }

  // walls used per model
  {
    const c = card("Walls used by model", "Walls placed per game by each model (0–10).", host);
    legend(c, [[LA, CA], [LB, CB]]);
    const usedBy = (s, m) => s.pb === m ? s.used.B : s.used.W;
    const ua = S.map(s => usedBy(s, A)), ub = S.map(s => usedBy(s, Bm));
    const bins = binsFor([ua, ub], 1, 0, WALLS_PER_PLAYER);
    mini(c, `mean ${LA} ${f1(describe(ua).mean)} · ${LB} ${f1(describe(ub).mean)}`);
    barChart(c, [{ name: LA, color: CA, values: bins.counts[0] }, { name: LB, color: CB, values: bins.counts[1] }], { labels: bins.labels, tickEvery: 1 });
  }

  // eval quality: each move's comment comes from the searching (moving) model
  {
    const c = card("Eval quality by model", "Each move's eval comes from the model that searched it. Here it is taken from the mover's own view: predicted own win% vs whether it actually won.", host);
    const acc = match.models.map(() => ({ n: 0, brier: 0, sumP: 0, wins: 0, bins: Array.from({ length: 10 }, () => ({ n: 0, wins: 0, sumP: 0 })) }));
    for (const s of S) for (const e of s.evs) {
      const m = mIdx(e.pla === "B" ? s.pb : s.pw); if (m < 0) continue;
      const p = e.pla === "W" ? e.p : 1 - e.p, y = s.winnerModel === match.models[m] ? 1 : 0, a = acc[m], b = a.bins[Math.min(9, Math.floor(p * 10))];
      a.n++; a.brier += (p - y) ** 2; a.sumP += p; a.wins += y; b.n++; b.wins += y; b.sumP += p;
    }
    const r = (a, lab) => `<tr><td>${lab}</td><td>${a.n}</td><td>${a.n ? (a.brier / a.n).toFixed(3) : "—"}</td><td>${a.n ? pct(a.sumP / a.n) : "—"}</td><td>${a.n ? pct(a.wins / a.n) : "—"}</td><td>${a.n ? ((a.sumP - a.wins) / a.n * 100 >= 0 ? "+" : "") + ((a.sumP - a.wins) / a.n * 100).toFixed(1) + "pt" : "—"}</td></tr>`;
    c.insertAdjacentHTML("beforeend", `<table class="st"><tr><th></th><th>positions</th><th>Brier</th><th>mean pred. own win%</th><th>actual own win%</th><th>over-confidence</th></tr>${r(acc[0], LA)}${r(acc[1], LB)}</table>`);
    mini(c, "Calibration: bar = actual own win rate among positions in each predicted bin (bins with < 20 positions left empty). Dashed = perfect.");
    legend(c, [[LA, CA], [LB, CB]]);
    const labels = acc[0].bins.map((_, k) => `${k * 10}–${k * 10 + 10}%`);
    const vals = acc.map(a => a.bins.map(b => b.n >= 20 ? b.wins / b.n : 0));
    const svg = barChart(c, [{ name: LA, color: CA, values: vals[0] }, { name: LB, color: CB, values: vals[1] }], { labels, yMax: 1, yFmt: v => Math.round(v * 100) + "%", tickEvery: 1,
      tickLabels: labels.map(l => l.split("–")[0]),
      tipFor: k => `<b>predicted ${labels[k]}</b>` + acc.map((a, m) => `<br>${sw(match.colors[m])}${match.labels[m]}: ${a.bins[k].n} pos · actual ${a.bins[k].n ? pct(a.bins[k].wins / a.bins[k].n) : "—"}`).join("") });
    const L = 40, band = (560 - 40 - 10) / 10, T = 16, ph = 170 - 16 - 26;
    el("line", { x1: L, y1: T + ph, x2: L + 10 * band, y2: T, stroke: "#3e4c59", "stroke-dasharray": "4 3", "stroke-width": 1.2, "pointer-events": "none" }, svg);
  }
}

