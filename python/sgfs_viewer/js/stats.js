"use strict";
// ---- Statistics ------------------------------------------------------------
function summarize(line, h) {
  const g = parseGame(line);
  const decisive = h.winner === "B" || h.winner === "W";
  const s = {
    i: h.i, n: g.moves.length, start: g.startTurnIdx, searched: g.moves.length - g.startTurnIdx,
    winner: h.winner, y: scoreB(h.winner), gtype: h.gtype,
    // result margin of a decided game (null for a draw, a resignation / forfeit (W+R, B+F) or an unknown result)
    margin: decisive && isFinite(parseFloat(h.result.slice(2))) ? Math.abs(parseFloat(h.result.slice(2))) : null,
    used: { B: 0, W: 0 }, hw: 0, vw: 0, pawnMoves: 0, evs: [], firstWallSearch: null,
    firstToMove: g.moves[g.startTurnIdx]?.pla ?? null,
    pb: h.pb, pw: h.pw, winnerModel: h.winnerModel, komi: h.komi, wb: h.wb, ww: h.ww, rulesLabel: h.rulesLabel,
    repetition: h.repetition, ruleKnown: h.ruleKnown, drawReason: h.drawReason, opening: h.opening,
  };
  s.signed = s.margin === null ? null : s.winner === "W" ? s.margin : -s.margin;
  s.walls = 0;
  // path race: shortest-path length to goal for both pawns after every move (no jumps)
  const bl = newBlock(), pos = { B: { ...g.startB }, W: { ...g.startW } };
  const dG = { B: goalDist(bl, "B"), W: goalDist(bl, "W") };
  const loser = s.winner === "B" ? "W" : "B";
  s.raceDeficit = -Infinity; s.maxPath = 0;
  const race = () => {
    const d = { B: dG.B[cellOf(pos.B)], W: dG.W[cellOf(pos.W)] };
    if (decisive) s.raceDeficit = Math.max(s.raceDeficit, d[s.winner] - d[loser]);
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
    if (hasEv(mv)) s.evs.push({ i, pla: mv.pla, p: mv.ev.wWin, score: mv.ev.wMargin, lead: mv.ev.hasLead, v: mv.ev.v, w: mv.ev.weight });
    race();
  });
  if (!isFinite(s.raceDeficit)) s.raceDeficit = null;
  // cycling: longest run of plies that repeat an earlier position; for a repetition draw, where the last run began
  const rs = repeatStretch(g);
  s.cycleMax = rs.maxRun; s.cycleStart = rs.maxStart; s.cycleEndStart = rs.endStart;
  // eval-based metrics, all over the searched positions, from the eventual winner's view where it matters.
  // Winner-based metrics are null for draws; the Brier score takes a draw's outcome as ½ (White's win prob in the
  // move comments counts a draw as ½).
  const E = s.evs, has = E.length > 0, hasW = has && decisive;
  const pw = e => s.winner === "W" ? e.p : 1 - e.p;
  const sw = e => s.winner === "W" ? e.score : -e.score;
  s.maxV = has ? Math.max(...E.map(e => e.v ?? 0)) : 0;   // this game's full-search visits
  s.minWinnerP = hasW ? Math.min(...E.map(pw)) : null;
  s.hasLead = has && E.every(e => e.lead);   // margin values below are the predicted lead (else the old score)
  s.minWinnerScore = hasW ? Math.min(...E.map(sw)) : null;
  const yW = s.y === null ? null : 1 - s.y;   // White's score
  let lastBelow = -1, lead = 0, vol = 0, swing = 0, brier = 0, prevSide = 0;
  E.forEach((e, k) => {
    if (hasW && pw(e) < 0.9) lastBelow = e.i;
    const side = e.p > 0.5 ? 1 : e.p < 0.5 ? -1 : 0;
    if (side && prevSide && side !== prevSide) lead++;
    if (side) prevSide = side;
    if (k) { const d = Math.abs(e.p - E[k - 1].p); vol += d; swing = Math.max(swing, d); }
    if (yW !== null) brier += (e.p - yW) ** 2;
  });
  s.decidedLeft = hasW ? s.n - (lastBelow + 1) : null;  // moves remaining once decided
  s.leadChanges = has ? lead : null;
  s.volatility = has ? vol : null;
  s.maxSwing = has ? swing : null;
  s.brier = has && yW !== null ? brier / E.length : null;
  return s;
}

// Black / draw / White counts and Black's score with a draw as ½. Games with an unknown result are left out.
function bdw(arr) {
  let b = 0, d = 0, w = 0;
  for (const s of arr) if (s.winner === "B") b++; else if (s.winner === "W") w++; else if (isDraw(s.winner)) d++;
  const n = b + d + w;
  return { b, d, w, n, score: n ? (b + d / 2) / n : NaN, drawRate: n ? d / n : NaN };
}
const DRAW_COLOR = "#c9c3b6";

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

function card(title, desc, parent = $("statCards"), wide = false) {
  const c = document.createElement("div");
  c.className = wide ? "card wide" : "card";
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

// Score bar: the shares of A wins / draws / B wins as three segments, with a tick at A's score (draw = ½).
function scoreRow(host, lab, sub, r, cA, cB, nameA, nameB) {
  const d = document.createElement("div");
  d.className = "wr-row";
  const a = r.n ? r.a / r.n : 0, dr = r.n ? r.d / r.n : 0, b = r.n ? r.b / r.n : 0;
  d.innerHTML = `<div class="lab">${lab}<small>${sub}</small></div><div><div class="wr-bar"><div style="flex-basis:${100 * a}%;background:${cA}"></div>` +
    (r.d ? `<div style="flex-basis:${100 * dr}%;background:${DRAW_COLOR}"></div>` : "") + `<div style="flex-basis:${100 * b}%;background:${cB}"></div>` +
    `<i class="score-tick" style="left:${100 * r.score}%"></i></div><div class="wr-nums"><span>${nameA} ${pct(a)}</span>` +
    (r.d ? `<span>draw ${pct(dr)}</span>` : "") + `<span>${nameB} ${pct(b)}</span></div></div>`;
  d.title = `${nameA} score (draw = ½): ${pct(r.score)}`;
  host.appendChild(d);
}
const bdwRow = (host, lab, arr) => {
  const r = bdw(arr);
  scoreRow(host, lab, `${r.n} games · Black score ${pct(r.score)}${r.d ? ` · ${r.d} draws` : ""}`, { a: r.b, d: r.d, b: r.w, n: r.n, score: r.score },
    "var(--b-acc)", "var(--w-acc)", "B", "W");
};
const table = (host, head, rows) => host.insertAdjacentHTML("beforeend",
  `<table class="st"><tr>${head.map(h => `<th>${h}</th>`).join("")}</tr>${rows.join("")}</table>`);
const tr = (cells, attrs = "") => `<tr${attrs}>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
const pctOr = (x, d = 1) => isFinite(x) ? pct(x, d) : "—";
const drawReasonCounts = arr => {
  const c = {};
  for (const s of arr) if (isDraw(s.winner)) { const r = s.drawReason || "unknown"; c[r] = (c[r] || 0) + 1; }
  return c;
};
const reasonText = c => Object.entries(c).map(([r, n]) => `${DRAW_REASONS[r] || r} ${n}`).join(" · ");
// Rule groups present in a set of games: N = 3, N = 4, ..., off, unknown (no RU).
function ruleGroups(S) {
  const ns = [...new Set(S.filter(s => s.repetition > 0).map(s => s.repetition))].sort();
  const out = ns.map(n => [`Rule on, N = ${n}`, S.filter(s => s.repetition === n)]);
  if (ns.length > 1) out.push(["Rule on, all", S.filter(s => s.repetition > 0)]);
  out.push(["Rule off", S.filter(s => s.ruleKnown && !s.repetition)]);
  out.push(["Rule not in file", S.filter(s => !s.ruleKnown)]);
  return out.filter(([, a]) => a.length);
}

function renderStats() {
  if (!games.length) { $("statsNote").textContent = "Load a file first."; return; }
  if (!statsCache) { $("statsNote").textContent = "Computing game metrics…"; return; }   // computeSummaries() re-renders when done
  // the statistics follow the game list's filters (one shared filter state)
  const S = filtered.map(i => statsCache[i]).sort((a, b) => a.i - b.i);
  const af = activeFilters();
  $("statTiles").innerHTML = ""; $("statCards").innerHTML = ""; $("matchTiles").innerHTML = ""; $("matchCards").innerHTML = "";
  $("statsNote").textContent = `${S.length} of ${games.length} games`;
  $("statsFilters").innerHTML = af.length ? `· filters (set in the Viewer's game list): <b>${af.map(escapeHtml).join(" · ")}</b> <button class="linkbtn" id="statsClear">Clear filters</button>` : "· no filters (set them in the Viewer's game list)";
  if ($("statsClear")) $("statsClear").onclick = clearFilters;
  $("matchSection").hidden = $("genTitle").hidden = !match;
  if (!S.length) return;
  if (match) renderMatchStats(S);
  const B = S.filter(s => s.winner === "B"), Wn = S.filter(s => s.winner === "W"), D = S.filter(s => isDraw(s.winner));
  const R = bdw(S), decided = [...B, ...Wn], withMargin = decided.filter(s => s.margin !== null);
  const C_B = "var(--b-acc)", C_W = "var(--w-acc)";
  const dAll = describe(S.map(s => s.n)), dB = describe(B.map(s => s.n)), dW = describe(Wn.map(s => s.n)), dD = describe(D.map(s => s.n));
  const dS = describe(S.map(s => s.searched));

  // calibration: a draw's outcome is ½ for White, as in the move comments
  const calib = Array.from({ length: 10 }, () => ({ n: 0, sumP: 0, wins: 0 }));
  let brier = 0, nPos = 0, fullV = 0, sumW = 0, nW = 0;
  const fullVs = new Map();
  for (const s of S) if (s.maxV) fullVs.set(s.maxV, (fullVs.get(s.maxV) || 0) + 1);
  for (const s of S) {
    if (s.y === null) continue;
    for (const e of s.evs) {
      const y = 1 - s.y, b = Math.min(9, Math.floor(e.p * 10));
      calib[b].n++; calib[b].sumP += e.p; calib[b].wins += y;
      brier += (e.p - y) ** 2; nPos++;
      if (e.v === s.maxV) fullV++;
      if (e.w !== undefined) { sumW += e.w; nW++; }
    }
  }
  const comeback = S.filter(s => s.minWinnerP !== null && s.minWinnerP < 0.2).length;
  const withEv = S.filter(s => s.minWinnerP !== null).length;
  const fullVText = [...fullVs.entries()].sort((a, b) => b[1] - a[1]).slice(0, 3).map(([v]) => v).join("/");
  const reasons = drawReasonCounts(S);

  const tiles = [
    ["Games", S.length, `${R.b} B · ${R.d} D · ${R.w} W`],
    ["Black (1st) score", pctOr(R.score), `draw = ½ · B ${pctOr(R.b / R.n, 0)} · D ${pctOr(R.drawRate, 0)} · W ${pctOr(R.w / R.n, 0)}`],
    ["Draws", pctOr(R.drawRate), R.d ? reasonText(reasons) : "none"],
    ["Game length", f1(dAll.mean), `± ${f1(dAll.sd)} · median ${f1(dAll.median)}`],
    ["Length range", `${dAll.min}–${dAll.max}`, `p10 ${f1(dAll.p10)} · p90 ${f1(dAll.p90)}`],
    ["Searched moves", f1(dS.mean), `± ${f1(dS.sd)} · median ${f1(dS.median)}`],
    ["Final margin", f1(describe(withMargin.map(s => s.margin)).mean), `median ${f1(describe(withMargin.map(s => s.margin)).median)} · decided games`],
    ["Walls used", `${f1(describe(S.map(s => s.used.B)).mean)} / ${f1(describe(S.map(s => s.used.W)).mean)}`, "mean per game, B / W"],
    ["Brier score", nPos ? (brier / nPos).toFixed(3) : "—", `${nPos} searched positions · draw = ½`],
    ["Comebacks", withEv ? pct(comeback / withEv) : "—", "winner once < 20% win · decided games"],
    ["Full searches", nPos ? pct(fullV / nPos) : "—", `v = the game's max visits (${fullVText || "—"}) · mean weight ${nW ? (sumW / nW).toFixed(2) : "—"}`],
  ];
  $("statTiles").innerHTML = tiles.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  // 0. repetition rule on / off side by side
  if (ruleGroups(S).length > 1) {
    const c = card("By repetition rule", "The main numbers for games with the repetition rule on (by N, the occurrence that draws) and off. Black score counts a draw as ½. Rule-off games cannot draw by repetition and run longer.", $("statCards"), true);
    const rows = ruleGroups(S).map(([lab, arr]) => {
      const r = bdw(arr), rc = drawReasonCounts(arr), dl = describe(arr.map(s => s.n)), ds = describe(arr.map(s => s.searched));
      const cyc = arr.filter(s => s.cycleMax >= 8).length;
      return tr([lab, arr.length, pctOr(r.score), pctOr(r.drawRate), rc.repetition || 0, rc.maxPlies || 0, f1(dl.mean), f1(dl.median), f1(ds.mean), pct(cyc / arr.length)]);
    });
    table(c, ["", "games", "Black score", "draws", "by rep.", "by ply limit", "mean len", "median len", "searched", "cycled ≥ 8"], rows);
  }

  // 1. komi table (walls 10/10, so the fence handicap does not mix in)
  {
    const tenTen = S.filter(s => s.wb === STANDARD_WALLS && s.ww === STANDARD_WALLS);
    const ks = [...new Set(tenTen.map(s => s.komi))].sort((a, b) => a - b);
    if (ks.length > 1) {
      const c = card("By komi", "Games with 10/10 walls, by komi (positive favours White; standard −0.5). Black score counts a draw as ½.");
      table(c, ["komi", "games", "B", "D", "W", "Black score", "draw rate", "mean len", "rule on"], ks.map(k => {
        const arr = tenTen.filter(s => s.komi === k), r = bdw(arr);
        return tr([fmtKomi(k) + (k === STANDARD_KOMI ? " (std)" : ""), arr.length, r.b, r.d, r.w, pctOr(r.score), pctOr(r.drawRate), f1(describe(arr.map(s => s.n)).mean), pct(arr.filter(s => s.repetition > 0).length / arr.length, 0)]);
      }));
    }
  }

  // 2. score by slice
  {
    const c = card("Black (1st) score by slice", "Shares of Black wins, draws (grey) and White wins; the tick and the label give Black's score with a draw as ½. Komi slices are in the komi table.");
    legend(c, [["Black (1st) wins", C_B], ["Draws", DRAW_COLOR], ["White (2nd) wins", C_W]]);
    const rows = [["All games", S]];
    const gts = [...new Set(S.map(s => s.gtype))].sort();
    if (gts.length > 1) for (const gt of gts) rows.push([`gtype = ${gt || "—"}`, S.filter(s => s.gtype === gt)]);
    for (const [lab, arr] of ruleGroups(S)) if (arr.length < S.length) rows.push([lab, arr]);
    rows.push(["Black moves first after init", S.filter(s => s.firstToMove === "B")]);
    rows.push(["White moves first after init", S.filter(s => s.firstToMove === "W")]);
    const q = [dAll.p25, dAll.median, dAll.p75];
    rows.push([`Short games (≤ ${f1(q[0])})`, S.filter(s => s.n <= q[0])]);
    rows.push([`Long games (> ${f1(q[2])})`, S.filter(s => s.n > q[2])]);
    // fence handicap: one row per initial-walls pair, so handicaps on Black and on White are never pooled
    const wallKeys = [...new Set(S.map(s => `${s.wb}/${s.ww}`))];
    if (wallKeys.length > 1) {
      const order = k => { const [b, w] = k.split("/").map(Number); return (b < w ? 0 : b > w ? 2 : 1) * 100 + Math.min(b, w); };
      for (const k of wallKeys.sort((a, b) => order(a) - order(b))) {
        const [wb, ww] = k.split("/").map(Number);
        rows.push([`Walls ${k}` + (wb < ww ? " (Black handicap)" : ww < wb ? " (White handicap)" : ""), S.filter(s => s.wb === wb && s.ww === ww)]);
      }
    }
    for (const [lab, arr] of rows) if (arr.length) bdwRow(c, lab, arr);
  }

  // 3. game length histogram
  {
    const c = card("Game length", "Total moves per game (including random init). Dashed lines: median and mean.");
    const w = niceWidth(dAll.max - dAll.min);
    const bins = binsFor([S.map(s => s.n)], w);
    mini(c, `mean ${f1(dAll.mean)} · median ${f1(dAll.median)} · sd ${f1(dAll.sd)} · IQR ${f1(dAll.p25)}–${f1(dAll.p75)}`);
    barChart(c, [{ name: "Games", color: "var(--neutral)", values: bins.counts[0] }], { labels: bins.labels, markers: markersFor(dAll, bins),
      tipFor: b => `<b>${bins.labels[b]} moves</b><br>${bins.counts[0][b]} games (${pct(bins.counts[0][b] / S.length)})` });
  }

  // 4. game length by result, small multiples on shared scales
  {
    const c = card("Game length by result", "Same bins and y-scale for all panels, so the shapes compare directly.");
    const w = niceWidth(dAll.max - dAll.min);
    const bins = binsFor([B.map(s => s.n), Wn.map(s => s.n), D.map(s => s.n)], w, Math.floor(dAll.min / w) * w, dAll.max);
    const yMax = Math.max(...bins.counts.flat(), 1);
    [["Black (1st) wins", C_B, 0, dB, B.length], ["White (2nd) wins", C_W, 1, dW, Wn.length], ["Draws", DRAW_COLOR, 2, dD, D.length]].forEach(([name, col, j, d, n]) => {
      if (j === 2 && !n) return;
      legend(c, [[`${name} · ${n} games`, col]]);
      mini(c, n ? `mean ${f1(d.mean)} · median ${f1(d.median)} · sd ${f1(d.sd)} · range ${d.min}–${d.max}` : "none");
      if (n) barChart(c, [{ name, color: col, values: bins.counts[j] }], { labels: bins.labels, yMax, height: 130, markers: markersFor(d, bins),
        tipFor: b => `<b>${bins.labels[b]} moves</b><br>${sw(col)}${bins.counts[j][b]} games (${pct(bins.counts[j][b] / n)} of ${name.toLowerCase()})` });
    });
  }

  // 5. draws
  if (D.length) {
    const c = card("Draws", "Why drawn games ended and how long they ran. Draw rates by rule and by komi are in the tables above.");
    const rows = Object.entries(reasons).map(([r, n]) => {
      const arr = D.filter(s => (s.drawReason || "unknown") === r), d = describe(arr.map(s => s.n));
      return tr([DRAW_REASONS[r] || r, n, pct(n / S.length), f1(d.mean), f1(d.median), `${d.min}–${d.max}`]);
    });
    table(c, ["reason", "games", "of all", "mean len", "median len", "range"], rows);
    const rep = D.filter(s => s.drawReason === "repetition" && s.cycleEndStart !== null);
    if (rep.length) {
      const vals = rep.map(s => s.cycleEndStart), d = describe(vals), loops = describe(rep.map(s => s.n - s.cycleEndStart + 1));
      mini(c, `<br>Repetition draws: the final repeated stretch began at ply mean ${f1(d.mean)} · median ${f1(d.median)} and lasted ${f1(loops.mean)} plies on average (median ${f1(loops.median)}).`);
      const bins = binsFor([vals], niceWidth(d.max - d.min), 0, d.max);
      barChart(c, [{ name: "Games", color: DRAW_COLOR, values: bins.counts[0] }], { labels: bins.labels, markers: markersFor(d, bins), height: 140,
        tipFor: b => `<b>stretch began at ply ${bins.labels[b]}</b><br>${bins.counts[0][b]} repetition draws` });
    }
  }

  // 6. cycling
  {
    const c = card("Cycling", "Longest repeated stretch per game: plies in a row whose position (pawns, walls placed, side to move) occurred before. Sort or filter the game list by it to find games that cycled. Rule-off games that cycled and still finished decisively broke out of the loop.");
    const groups = ruleGroups(S).filter(([lab]) => lab !== "Rule on, all");
    table(c, ["", "games", "any repeat", "≥ 4 plies", "≥ 8 plies", "≥ 16 plies", "≥ 8 and decided", "mean longest"], groups.map(([lab, arr]) => {
      const f = k => pct(arr.filter(s => s.cycleMax >= k).length / arr.length);
      return tr([lab, arr.length, f(1), f(4), f(8), f(16), arr.filter(s => s.cycleMax >= 8 && (s.winner === "B" || s.winner === "W")).length, f1(describe(arr.map(s => s.cycleMax)).mean)]);
    }));
  }

  // 7. frequent openings
  {
    const by = new Map();
    for (const s of S) { const a = by.get(s.opening); a ? a.push(s) : by.set(s.opening, [s]); }
    const top = [...by.entries()].sort((a, b) => b[1].length - a[1].length).slice(0, 12);
    if (top.length > 1 || (top.length === 1 && S.length > 1)) {
      const c = card("Frequent openings", `The first ${OPENING_PLIES} plies (random init plies included) of the most common lines. Click a row to filter the game list to that line.`);
      table(c, ["line", "games", "share", "Black score", "draws", "mean len"], top.map(([key, arr]) => {
        const r = bdw(arr);
        return tr([`<span class="line">${escapeHtml(openingLabel(key))}</span>`, arr.length, pct(arr.length / S.length), pctOr(r.score), pctOr(r.drawRate), f1(describe(arr.map(s => s.n)).mean)], ` class="click" data-open="${escapeHtml(key)}"`);
      }));
      mini(c, `${by.size} distinct lines in ${S.length} games.`);
      c.querySelector("table").addEventListener("click", e => {
        const row = e.target.closest("tr[data-open]");
        if (row) { setOpeningFilter(row.dataset.open); setTab(false); }
      });
    }
  }

  // 8. Black score by length bin
  {
    const c = card("Black (1st) score vs game length", "Per length bin, with a draw as ½; bins with fewer than 10 games are left out. Dashed line: 50%.");
    const w = niceWidth(dAll.max - dAll.min, 20);
    const lo = Math.floor(dAll.min / w) * w;
    const bins = binsFor([S.map(s => s.n)], w, lo, dAll.max);
    const grp = bins.counts[0].map(() => []);
    for (const s of S) grp[Math.min(bins.nb - 1, Math.floor((s.n - lo) / w))].push(s);
    const rs = grp.map(bdw), rate = rs.map(r => r.n >= 10 ? r.score : 0);
    const svg = barChart(c, [{ name: "Black score", color: C_B, values: rate }], { labels: bins.labels, yMax: 1, yFmt: v => Math.round(v * 100) + "%",
      tipFor: k => `<b>${bins.labels[k]} moves</b><br>${rs[k].n} games` + (rs[k].n >= 10 ? `<br>${sw(C_B)}Black score ${pct(rate[k])}<br>B ${rs[k].b} · D ${rs[k].d} · W ${rs[k].w}` : "<br>(too few games)") });
    const y = 16 + (170 - 16 - 26) * 0.5;
    el("line", { x1: 40, x2: 550, y1: y, y2: y, stroke: "#3e4c59", "stroke-dasharray": "4 3", "stroke-width": 1 }, svg);
  }

  // 9. signed final margin
  if (withMargin.length) {
    const c = card("Final margin (White's view)", "Result margin (the SGF result: since the I/O v2 rules the tempo lead, e.g. W+0.5); negative = Black won by that much, positive = White won. Draws and resignations / forfeits are left out.");
    legend(c, [["Black wins", C_B], ["White wins", C_W]]);
    const vals = withMargin.map(s => s.signed), lo = Math.min(...vals), hi = Math.max(...vals);
    const w = niceWidth(hi - lo, 40);
    const bins = binsFor([vals], w);
    const d = describe(vals);
    mini(c, `mean ${fmtScore(d.mean)} · median ${fmtScore(d.median)} · mean |margin| B wins ${f1(describe(B.filter(s => s.margin !== null).map(s => s.margin)).mean)}, W wins ${f1(describe(Wn.filter(s => s.margin !== null).map(s => s.margin)).mean)}` + (S.length > withMargin.length ? ` · ${S.length - withMargin.length} draws / resignations left out` : ""));
    barChart(c, [{ name: "Games", color: C_W, values: bins.counts[0] }], { labels: bins.labels,
      colorsFor: b => bins.lo + b * w < 0 ? C_B : C_W,
      tipFor: b => `<b>margin ${bins.labels[b]}</b><br>${bins.counts[0][b]} games` });
  }

  // 10. walls used by winner vs loser
  if (decided.length) {
    const c = card("Walls used: winner vs loser", "Walls placed per game (0–10) by the winning and the losing side. Draws are left out.");
    legend(c, [["Winner", "var(--margin)"], ["Loser", "var(--neutral)"]]);
    const win = decided.map(s => s.used[s.winner]), lose = decided.map(s => s.used[s.winner === "B" ? "W" : "B"]);
    const bins = binsFor([win, lose], 1, 0, WALLS_PER_PLAYER);
    const hv = S.reduce((a, s) => [a[0] + s.hw, a[1] + s.vw], [0, 0]);
    mini(c, `mean winner ${f1(describe(win).mean)} · loser ${f1(describe(lose).mean)} · horizontal ${pct(hv[0] / (hv[0] + hv[1] || 1))} of walls`);
    barChart(c, [{ name: "Winner", color: "var(--margin)", values: bins.counts[0] }, { name: "Loser", color: "var(--neutral)", values: bins.counts[1] }], { labels: bins.labels, tickEvery: 1 });
  }

  // 11. walls by side
  {
    const c = card("Walls used by side", "Walls placed per game by Black (1st) and White (2nd).");
    legend(c, [["Black (1st)", C_B], ["White (2nd)", C_W]]);
    const bins = binsFor([S.map(s => s.used.B), S.map(s => s.used.W)], 1, 0, WALLS_PER_PLAYER);
    barChart(c, [{ name: "Black", color: C_B, values: bins.counts[0] }, { name: "White", color: C_W, values: bins.counts[1] }], { labels: bins.labels, tickEvery: 1 });
  }

  // 12. calibration
  if (nPos) {
    const c = card("Win% calibration", "Searched positions binned by predicted White win%; bar = White's actual score in that bin (a draw counts as ½, as in the predictions). Dashed = perfect calibration.");
    const labels = calib.map((_, k) => `${k * 10}–${k * 10 + 10}%`);
    const actual = calib.map(b => b.n ? b.wins / b.n : 0);
    const svg = barChart(c, [{ name: "Actual White score", color: C_W, values: actual }], { labels, yMax: 1, yFmt: v => Math.round(v * 100) + "%", tickEvery: 1,
      tickLabels: labels.map(l => l.split("–")[0]),
      tipFor: k => `<b>predicted ${labels[k]}</b><br>${calib[k].n} positions<br>mean predicted ${calib[k].n ? pct(calib[k].sumP / calib[k].n) : "—"}<br>${sw(C_W)}actual score ${calib[k].n ? pct(actual[k]) : "—"}` });
    const L = 40, band = (560 - 40 - 10) / 10, T = 16, ph = 170 - 16 - 26;
    el("line", { x1: L, y1: T + ph, x2: L + 10 * band, y2: T, stroke: "#3e4c59", "stroke-dasharray": "4 3", "stroke-width": 1.2, "pointer-events": "none" }, svg);
  }

  // 13. how early is the game decided
  {
    const c = card("Moves left after the game is decided", "Moves from the last search where the eventual winner was below 90% until the end. Draws are left out.");
    const vals = S.filter(s => s.decidedLeft !== null).map(s => s.decidedLeft);
    if (vals.length) {
      const d = describe(vals), w = niceWidth(d.max - d.min);
      const bins = binsFor([vals], w, 0, d.max);
      mini(c, `mean ${f1(d.mean)} · median ${f1(d.median)} · sd ${f1(d.sd)}`);
      barChart(c, [{ name: "Games", color: "var(--margin)", values: bins.counts[0] }], { labels: bins.labels, markers: markersFor(d, bins),
        tipFor: b => `<b>${bins.labels[b]} moves left</b><br>${bins.counts[0][b]} games` });
    }
  }

  // 14. random init length + first searched wall
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

  // 15. summary table
  {
    const c = card("Length summary table", "Total moves per game.");
    const row = (lab, d) => `<tr><td>${lab}</td><td>${d.n}</td><td>${f1(d.mean)}</td><td>${f1(d.sd)}</td><td>${d.min ?? "—"}</td><td>${f1(d.p25)}</td><td>${f1(d.median)}</td><td>${f1(d.p75)}</td><td>${d.max ?? "—"}</td></tr>`;
    c.insertAdjacentHTML("beforeend", `<table class="st"><tr><th></th><th>n</th><th>mean</th><th>sd</th><th>min</th><th>p25</th><th>median</th><th>p75</th><th>max</th></tr>` +
      row("All", dAll) + row("Black wins", dB) + row("White wins", dW) + (D.length ? row("Draws", dD) : "") + row("Searched only", dS) + `</table>`);
  }
}

// ---- Match (two different models) statistics -------------------------------
// Every rate here is a score with a draw as ½ point. The Wilson interval and the p-value treat the score as a
// binomial proportion (draws count as half a win), a slight approximation when there are draws.
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

// Wins / draws / losses of model m in a set of games, and its score (draw = ½); unknown results are left out.
function wdlOf(arr, m) {
  let w = 0, d = 0, l = 0;
  for (const s of arr) if (isDraw(s.winner)) d++; else if (s.winnerModel === m) w++; else if (s.winnerModel) l++;
  const n = w + d + l, pts = w + d / 2;
  return { w, d, l, n, pts, score: n ? pts / n : NaN };
}
const fmtPts = x => Number.isInteger(x) ? String(x) : x.toFixed(1);

function renderMatchStats(S) {
  const [A, Bm] = match.models, [LA, LB] = match.labels, [CA, CB] = match.colors;
  const host = $("matchCards");
  $("matchTitle").innerHTML = `Match · ${escapeHtml(LA)} vs ${escapeHtml(LB)} <small>${escapeHtml(LA)} = ${escapeHtml(A)} · ${escapeHtml(LB)} = ${escapeHtml(Bm)}` +
    (match.labels[0] === "New" ? " · newer = more training samples (-s)" : "") + ` · scores count a draw as ½</small>`;
  const aB = S.filter(s => s.pb === A), aW = S.filter(s => s.pw === A);
  const T = wdlOf(S, A), [lo, hi] = wilson(T.pts, T.n), RB = wdlOf(aB, A), RW = wdlOf(aW, A), C = bdw(S);
  const pv = pTwoSided(T.pts, T.n);
  const tiles = [
    ["Games", S.length, `${LA} as Black ${aB.length} · as White ${aW.length}`],
    [`${LA} score`, `${fmtPts(T.pts)}–${fmtPts(T.n - T.pts)}`, `${pctOr(T.score)} · 95% CI ${pct(lo)}–${pct(hi)}`],
    [`${LA} W / D / L`, `${T.w} / ${T.d} / ${T.l}`, `draw rate ${pctOr(T.d / T.n)}`],
    [`Elo ${LA} − ${LB}`, fmtElo(elo(T.score)), `95% CI ${fmtElo(elo(lo))} … ${fmtElo(elo(hi))}`],
    ["p vs 50%", pv < 0.001 ? "<0.001" : pv.toFixed(3), pv < 0.05 ? "significant at 5%" : "not significant at 5%"],
    [`${LA} as Black (1st)`, `${fmtPts(RB.pts)}/${RB.n}`, pctOr(RB.score)],
    [`${LA} as White (2nd)`, `${fmtPts(RW.pts)}/${RW.n}`, pctOr(RW.score)],
    ["Black (1st) score", pctOr(C.score), "color bias in this match"],
  ];
  $("matchTiles").innerHTML = tiles.map(([k, v, s]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");

  // head-to-head
  {
    const c = card("Head-to-head", `Shares of games won by ${LA}, drawn (grey) and won by ${LB}, overall and by slice; the tick is ${LA}'s score with a draw as ½.`, host);
    legend(c, [[LA, CA], ["Draws", DRAW_COLOR], [LB, CB]]);
    const d = describe(S.map(s => s.n));
    const rows = [["All games", S], [`${LA} as Black (1st)`, aB], [`${LA} as White (2nd)`, aW],
      [`Short games (≤ ${f1(d.p25)})`, S.filter(s => s.n <= d.p25)], [`Long games (> ${f1(d.p75)})`, S.filter(s => s.n > d.p75)]];
    for (const gt of [...new Set(S.map(s => s.gtype))].sort()) if (new Set(S.map(s => s.gtype)).size > 1) rows.push([`gtype = ${gt}`, S.filter(s => s.gtype === gt)]);
    for (const [lab, arr] of rows) if (arr.length) {
      const r = wdlOf(arr, A), [l, h] = wilson(r.pts, r.n);
      scoreRow(c, lab, `${r.n} games · ${LA} score ${pctOr(r.score)} · CI ${pct(l, 0)}–${pct(h, 0)}`, { a: r.w, d: r.d, b: r.l, n: r.n, score: r.score }, CA, CB, LA, LB);
    }
  }

  // record table
  {
    const c = card("Record by color", `Rows are from ${LA}'s side. Score and Elo count a draw as ½. Margin = mean result margin of the games won.`, host);
    const row = (lab, arr) => {
      const r = wdlOf(arr, A);
      const mm = a => { a = a.filter(s => s.margin !== null); return a.length ? f1(describe(a.map(s => s.margin)).mean) : "—"; };
      return tr([lab, arr.length, r.w, r.d, r.l, pctOr(r.score), r.n ? fmtElo(elo(r.score)) : "—", arr.length ? f1(describe(arr.map(s => s.n)).mean) : "—",
        mm(arr.filter(s => s.winnerModel === A)), mm(arr.filter(s => s.winnerModel && s.winnerModel !== A))]);
    };
    table(c, ["", "games", `${LA} W`, "D", `${LA} L`, "score", "Elo", "mean len", "margin (won)", "margin (lost)"],
      [row(`${LA} as Black`, aB), row(`${LA} as White`, aW), row("Total", S)]);
  }

  // length by result
  {
    const c = card("Game length by result", "Same bins and y-scale for all panels.", host);
    const WA = S.filter(s => s.winnerModel === A), WB = S.filter(s => s.winnerModel === Bm), DD = S.filter(s => isDraw(s.winner));
    const all = S.map(s => s.n), lo = Math.min(...all), hi = Math.max(...all), w = niceWidth(hi - lo, 30);
    const bins = binsFor([WA.map(s => s.n), WB.map(s => s.n), DD.map(s => s.n)], w, Math.floor(lo / w) * w, hi);
    const yMax = Math.max(1, ...bins.counts.flat());
    [[`${LA} wins`, CA, 0, WA], [`${LB} wins`, CB, 1, WB], ["Draws", DRAW_COLOR, 2, DD]].forEach(([name, col, j, arr]) => {
      if (j === 2 && !arr.length) return;
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
    const decided = S.filter(s => s.winnerModel && s.margin !== null), draws = S.length - decided.length;
    if (decided.length) {
      const c = card(`Final margin (${LA}'s view)`, `Negative = ${LB} won by that much, positive = ${LA} won. Draws and resignations are left out.`, host);
      legend(c, [[`${LB} wins`, CB], [`${LA} wins`, CA]]);
      const vals = decided.map(s => s.winnerModel === A ? s.margin : -s.margin), lo = Math.min(...vals), hi = Math.max(...vals);
      const w = niceWidth(hi - lo, 40), bins = binsFor([vals], w), d = describe(vals);
      mini(c, `mean ${fmtScore(d.mean)} · median ${fmtScore(d.median)} · sd ${f1(d.sd)}` + (draws ? ` · ${draws} draws / resignations left out` : ""));
      barChart(c, [{ name: "Games", color: CA, values: bins.counts[0] }], { labels: bins.labels,
        colorsFor: b => bins.lo + b * w < 0 ? CB : CA, tipFor: b => `<b>margin ${bins.labels[b]}</b><br>${bins.counts[0][b]} games` });
    }
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
    const c = card("Eval quality by model", "Each move's eval comes from the model that searched it. Here it is taken from the mover's own view: predicted own win% vs its actual score (a draw counts as ½, as in the predictions).", host);
    const acc = match.models.map(() => ({ n: 0, brier: 0, sumP: 0, wins: 0, bins: Array.from({ length: 10 }, () => ({ n: 0, wins: 0, sumP: 0 })) }));
    for (const s of S) {
      if (s.y === null) continue;
      for (const e of s.evs) {
        const m = mIdx(e.pla === "B" ? s.pb : s.pw); if (m < 0) continue;
        const p = e.pla === "W" ? e.p : 1 - e.p, y = e.pla === "B" ? s.y : 1 - s.y, a = acc[m], b = a.bins[Math.min(9, Math.floor(p * 10))];
        a.n++; a.brier += (p - y) ** 2; a.sumP += p; a.wins += y; b.n++; b.wins += y; b.sumP += p;
      }
    }
    const r = (a, lab) => `<tr><td>${lab}</td><td>${a.n}</td><td>${a.n ? (a.brier / a.n).toFixed(3) : "—"}</td><td>${a.n ? pct(a.sumP / a.n) : "—"}</td><td>${a.n ? pct(a.wins / a.n) : "—"}</td><td>${a.n ? ((a.sumP - a.wins) / a.n * 100 >= 0 ? "+" : "") + ((a.sumP - a.wins) / a.n * 100).toFixed(1) + "pt" : "—"}</td></tr>`;
    c.insertAdjacentHTML("beforeend", `<table class="st"><tr><th></th><th>positions</th><th>Brier</th><th>mean pred. own win%</th><th>actual own score</th><th>over-confidence</th></tr>${r(acc[0], LA)}${r(acc[1], LB)}</table>`);
    mini(c, "Calibration: bar = actual own score among positions in each predicted bin (bins with < 20 positions left empty). Dashed = perfect.");
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
