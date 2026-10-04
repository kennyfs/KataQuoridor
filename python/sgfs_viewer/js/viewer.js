"use strict";
// ---- Rendering: board ------------------------------------------------------
// pawn-grid col/row (0..8) -> pixel top-left, honoring flip (180° rotation)
const vc = c => flipped ? N - 1 - c : c;
const px = c => PAD + vc(c) * PITCH;
const py = r => PAD + vc(r) * PITCH;

function drawBoard(st, paths) {
  const svg = $("board");
  const W = BOARD_PX + 2 * PAD;
  svg.setAttribute("viewBox", `0 0 ${W} ${W}`);
  svg.innerHTML = "";
  const defs = el("defs", {}, svg);
  const f = el("filter", { id: "shadow", x: "-30%", y: "-30%", width: "160%", height: "160%" }, defs);
  el("feDropShadow", { dx: 0, dy: 1.5, stdDeviation: 1.6, "flood-color": "#000", "flood-opacity": 0.35 }, f);
  const glow = el("filter", { id: "glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
  el("feDropShadow", { dx: 0, dy: 0, stdDeviation: 3.5, "flood-color": "#fff", "flood-opacity": 0.9 }, glow);

  el("rect", { x: 4, y: 4, width: W - 8, height: W - 8, rx: 22, fill: "var(--board)", stroke: "var(--board-edge)", "stroke-width": 3 }, svg);

  for (let i = 0; i < N; i++) {
    txt(svg, px(i) + CELL / 2, PAD - 10, String.fromCharCode(97 + i), { "text-anchor": "middle", "font-size": 12, fill: "var(--label)", "font-weight": 600 });
    txt(svg, PAD - 13, py(i) + CELL / 2 + 4, i + 1, { "text-anchor": "middle", "font-size": 12, fill: "var(--label)", "font-weight": 600 });
  }
  // goal rows tinted: Black heads to row 1 (y=0), White to row 9 (y=16)
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const fill = r === 0 ? "var(--cell-goal-b)" : r === N - 1 ? "var(--cell-goal-w)" : "var(--cell)";
    el("rect", { x: px(c), y: py(r), width: CELL, height: CELL, rx: 8, fill }, svg);
  }
  if (!cur) return;

  // all shortest paths: tinted cells + the shortest-path DAG, Black offset up-left, White down-right
  if (showPaths && paths) {
    for (const p of ["B", "W"]) {
      const col = p === "B" ? "var(--b-acc)" : "var(--w-acc)";
      for (const k of paths[p].cells)
        el("rect", { x: px(k % N) + 3, y: py((k / N) | 0) + 3, width: CELL - 6, height: CELL - 6, rx: 6, fill: col, opacity: 0.2 }, svg);
    }
    for (const p of ["B", "W"]) {
      const col = p === "B" ? "var(--b-acc)" : "var(--w-acc)", o = p === "B" ? -4 : 4;
      const ctr = k => [px(k % N) + CELL / 2 + o, py((k / N) | 0) + CELL / 2 + o];
      const d = paths[p].edges.map(([a, b]) => { const [x1, y1] = ctr(a), [x2, y2] = ctr(b); return `M${x1} ${y1}L${x2} ${y2}`; }).join("");
      if (d) el("path", { d, stroke: col, "stroke-width": 3, "stroke-linecap": "round", fill: "none", opacity: 0.75 }, svg);
      for (const k of paths[p].cells) if (((k / N) | 0) === GOAL[p]) { const [x, y] = ctr(k); el("circle", { cx: x, cy: y, r: 3.5, fill: col }, svg); }
    }
  }

  for (const w of st.walls) {
    const c = w.x >> 1, r = w.y >> 1;
    let x, y, wd, ht;
    if (w.kind === "hwall") {         // center (odd,odd): between rows r and r+1, spanning cols c, c+1
      x = px(flipped ? c + 1 : c) - 1; y = py(flipped ? r + 1 : r) + CELL + 1; wd = 2 * CELL + GAP + 2; ht = GAP - 2;
    } else {                            // upper arm (odd,even): between cols c and c+1, spanning rows r, r+1
      x = px(flipped ? c + 1 : c) + CELL + 1; y = py(flipped ? r + 1 : r) - 1; wd = GAP - 2; ht = 2 * CELL + GAP + 2;
    }
    const isLast = w.idx === ply - 1;
    el("rect", {
      x, y, width: wd, height: ht, rx: 4,
      fill: w.pla === "B" ? "var(--b-acc)" : "var(--w-acc)",
      stroke: isLast ? "#fff" : "rgba(0,0,0,.25)", "stroke-width": isLast ? 2 : 1,
      filter: isLast ? "url(#glow)" : "url(#shadow)",
    }, svg);
  }

  if (st.prevPawn && st.last) {
    const cx = px(st.prevPawn.x >> 1) + CELL / 2, cy = py(st.prevPawn.y >> 1) + CELL / 2;
    el("circle", { cx, cy, r: CELL * 0.33, fill: "none", stroke: st.last.pla === "B" ? "var(--b-acc)" : "var(--w-acc)", "stroke-width": 2, "stroke-dasharray": "4 4", opacity: 0.8 }, svg);
  }

  for (const p of ["B", "W"]) {
    const pos = st[p];
    const cx = px(pos.x >> 1) + CELL / 2, cy = py(pos.y >> 1) + CELL / 2;
    const isLast = st.last && moveKind(st.last) === "pawn" && st.last.pla === p;
    if (isLast) el("circle", { cx, cy, r: CELL * 0.44, fill: "none", stroke: p === "B" ? "var(--b-acc)" : "var(--w-acc)", "stroke-width": 2, opacity: 0.55 }, svg);
    el("circle", {
      cx, cy, r: CELL * 0.34,
      fill: p === "B" ? "var(--b-pawn)" : "var(--w-pawn)",
      stroke: p === "B" ? "var(--b-acc)" : "var(--w-acc)", "stroke-width": 4, filter: "url(#shadow)",
    }, svg);
    el("circle", { cx: cx - CELL * 0.1, cy: cy - CELL * 0.11, r: CELL * 0.09, fill: "#fff", opacity: p === "B" ? 0.18 : 0.8 }, svg);
  }
}

// One pip per initial wall (WB / WW, 10 unless the game has a fence handicap).
function drawWallsCounter(id, used, initial) {
  const box = $(id);
  box.innerHTML = "";
  for (let i = 0; i < initial; i++) {
    const s = document.createElement("i");
    if (i >= initial - used) s.className = "used";
    box.appendChild(s);
  }
}

// ---- Eval strip (search at the displayed position) -------------------------
function prevEvIdx(i) { for (let j = i - 1; j >= 0; j--) if (hasEv(cur.moves[j])) return j; return -1; }

function renderPathTile(st, paths) {
  const b = paths.B.len, w = paths.W.len;
  if (b < 0 || w < 0) { $("evPathV").textContent = "—"; $("evPathS").textContent = "no path?"; return; }
  const d = b - w;
  $("evPathV").innerHTML = `${d > 0 ? "+" : d < 0 ? "−" : "±"}${Math.abs(d)}<small>B ${b} · W ${w}</small>`;
  const over = ply >= cur.moves.length;
  $("evPathS").textContent = (d > 0 ? "White closer" : d < 0 ? "Black closer" : "Even race") + (over ? "" : ` · ${st.toMove === "B" ? "Black" : "White"} to move`) +
    ` · ${paths.B.cells.size}/${paths.W.cells.size} cells on paths`;
}

function renderEvalStrip() {
  const mv = cur.moves[ply];
  const tiles = ["evWin", "evMar", "evVis"];
  if (!hasEv(mv)) {
    tiles.forEach(t => $(t).classList.add("dim"));
    const why = ply >= cur.moves.length ? `Game over · ${resultLabel(games[curGameIdx])}` : ply < cur.startTurnIdx ? "Random init move, no search" : "No search data";
    $("evWinV").textContent = "—"; $("evMarV").textContent = "—"; $("evVisV").textContent = "—";
    $("evWinBar").style.flexBasis = "50%";
    $("evWinS").textContent = why; $("evMarS").textContent = ""; $("evVisS").textContent = "";
    return;
  }
  tiles.forEach(t => $(t).classList.remove("dim"));
  const ev = mv.ev, pj = prevEvIdx(ply), pev = pj >= 0 ? cur.moves[pj].ev : null;
  const dW = pev ? ev.wWin - pev.wWin : null, dS = pev ? ev.wMargin - pev.wMargin : null;
  const dSpan = (d, fmt) => d === null ? "" : ` <span class="delta">${d >= 0 ? "▲" : "▼"} ${fmt(Math.abs(d))}</span> vs prev`;
  $("evWinV").textContent = pct(ev.wWin);
  $("evWinBar").style.flexBasis = (100 * ev.wWin).toFixed(1) + "%";
  $("evWinS").innerHTML = `B ${pct(ev.wLoss)}` + (ev.noRes > 0 ? ` · draw ${pct(ev.noRes)}` : "") + " ·" + dSpan(dW, x => (100 * x).toFixed(1) + "pt");
  $("evMar").querySelector(".k").textContent = ev.hasLead ? "Predicted lead (W)" : "Margin (W) · no lead in file";
  $("evMarV").textContent = fmtScore(ev.wMargin);
  $("evMarS").innerHTML = (ev.wMargin >= 0 ? "White ahead" : "Black ahead") + (dS === null ? "" : " ·" + dSpan(dS, x => x.toFixed(1)));
  $("evVisV").textContent = ev.v ?? "—";
  $("evVisS").innerHTML = [ev.rv !== undefined ? `reanalysis ${ev.rv}` : null, ev.weight !== undefined ? `weight ${ev.weight.toFixed(2)}` : null, `chose <b>${mv.pla} ${notation(mv)}</b>`].filter(Boolean).join(" · ");
}

// ---- Rendering: side panels ------------------------------------------------
function renderInfo() {
  const g = cur, h = games[curGameIdx];
  const rows = [
    ["Game", `#${curGameIdx + 1} of ${games.length}`],
    ["Result", resultLabel(h)],
    ...(h.ruleKnown || h.repetition ? [["Repetition rule", h.repetition ? `on: draw at occurrence ${h.repetition} of a position` : "off"]] : []),
    ...(h.komi !== STANDARD_KOMI ? [["Komi", `${fmtKomi(h.komi)} (standard −0.5)`]] : []),
    ...(h.wb !== STANDARD_WALLS || h.ww !== STANDARD_WALLS ? [["Initial walls", `Black ${h.wb} · White ${h.ww}`]] : []),
    ["Moves", `${g.moves.length}  (init ${g.startTurnIdx}, searched ${g.moves.length - g.startTurnIdx})`],
    ["Type", (g.kv.gtype || "—") + (g.kv.usedInitialPosition ? " · initial pos" : "")],
    ["Players", (g.props.PB?.[0] || "—") + (g.props.PW?.[0] && g.props.PW[0] !== g.props.PB?.[0] ? " vs " + g.props.PW[0] : "")],
    ["Hash", g.kv.gameHash || "—"],
  ];
  $("info").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${escapeHtml(v)}</dd>`).join("");
}

function renderMoveList() {
  const parts = [];
  cur.moves.forEach((mv, i) => {
    if (i === 0 && cur.startTurnIdx > 0) parts.push(`<div class="sep">Random init · ${cur.startTurnIdx} moves</div>`);
    if (i === cur.startTurnIdx && i > 0) parts.push(`<div class="sep">Self-play search</div>`);
    const e = mv.ev;
    const ev = hasEv(mv) ? `W ${(e.wWin * 100).toFixed(0)}% · ${e.hasLead ? "lead" : "margin"} ${fmtScore(e.wMargin)} · v=${e.v ?? "?"}${e.result ? " · " + e.result : ""}` : (e?.result || "");
    parts.push(`<div class="m${i < cur.startTurnIdx ? " init" : ""}" data-i="${i + 1}"><span class="n">${i + 1}</span><span class="who ${mv.pla}"></span><span class="mv">${notation(mv)}</span><span class="ev">${ev}</span></div>`);
  });
  $("moveList").innerHTML = parts.join("");
}

// Search trace: two stacked panels sharing the move axis (win% on top, margin below),
// never a dual axis. Point i = search at position i (before move i+1).
const CH = { l: 34, r: 10, t: 8, h1: 96, gap: 18, h2: 80, b: 20 };
function chartGeom() {
  const svg = $("chart");
  const w = svg.clientWidth || 320, n = Math.max(1, cur.moves.length);
  const X = i => CH.l + (i / n) * (w - CH.l - CH.r);
  const top2 = CH.t + CH.h1 + CH.gap;
  return { svg, w, n, X, top2 };
}

const curHasLead = () => cur.moves.some(mv => hasEv(mv) && mv.ev.hasLead);

function renderChart() {
  $("marLegend").textContent = curHasLead() ? "Predicted lead" : "Margin (no lead in file)";
  const { svg, w, n, X, top2 } = chartGeom();
  const H = top2 + CH.h2 + CH.b;
  svg.setAttribute("viewBox", `0 0 ${w} ${H}`);
  svg.style.height = H + "px";
  svg.innerHTML = "";
  const Yw = v => CH.t + (1 - v) * CH.h1;
  const pts = [];
  cur.moves.forEach((mv, i) => { if (hasEv(mv)) pts.push({ i, ev: mv.ev }); });
  let M = 5;
  for (const p of pts) M = Math.max(M, Math.abs(p.ev.wMargin));
  M = Math.ceil(M / 5) * 5;
  const Ym = v => top2 + (1 - (v + M) / (2 * M)) * CH.h2;

  // init shading + every-10-move grid lines across both panels
  if (cur.startTurnIdx > 0) {
    el("rect", { x: X(0), y: CH.t, width: X(cur.startTurnIdx) - X(0), height: CH.h1, fill: "#f1eee7" }, svg);
    el("rect", { x: X(0), y: top2, width: X(cur.startTurnIdx) - X(0), height: CH.h2, fill: "#f1eee7" }, svg);
  }
  const labelEvery = n > 150 ? 50 : n > 70 ? 20 : 10;
  for (let i = 10; i < n; i += 10) {
    el("line", { x1: X(i), x2: X(i), y1: CH.t, y2: CH.t + CH.h1, stroke: "var(--grid)", "stroke-width": 1 }, svg);
    el("line", { x1: X(i), x2: X(i), y1: top2, y2: top2 + CH.h2, stroke: "var(--grid)", "stroke-width": 1 }, svg);
    if (i % labelEvery === 0) txt(svg, X(i), top2 + CH.h2 + 14, i, { "text-anchor": "middle" });
  }
  // baselines & y labels
  el("line", { x1: CH.l, x2: w - CH.r, y1: Yw(0.5), y2: Yw(0.5), stroke: "#d6d1c5", "stroke-dasharray": "3 3" }, svg);
  el("line", { x1: CH.l, x2: w - CH.r, y1: Ym(0), y2: Ym(0), stroke: "#d6d1c5", "stroke-dasharray": "3 3" }, svg);
  txt(svg, CH.l - 5, Yw(1) + 8, "100%", { "text-anchor": "end" });
  txt(svg, CH.l - 5, Yw(0.5) + 4, "50%", { "text-anchor": "end" });
  txt(svg, CH.l - 5, Yw(0), "0%", { "text-anchor": "end" });
  txt(svg, CH.l - 5, Ym(M) + 8, "+" + M, { "text-anchor": "end" });
  txt(svg, CH.l - 5, Ym(0) + 4, "0", { "text-anchor": "end" });
  txt(svg, CH.l - 5, Ym(-M), "−" + M, { "text-anchor": "end" });
  txt(svg, w - CH.r, CH.t + 10, "win%", { "text-anchor": "end", fill: "#6b7885" });
  txt(svg, w - CH.r, top2 + 10, curHasLead() ? "lead" : "margin", { "text-anchor": "end", fill: "#6b7885" });

  if (pts.length) {
    const line = (Y, key) => pts.map((p, k) => (k ? "L" : "M") + X(p.i).toFixed(1) + " " + Y(p.ev[key]).toFixed(1)).join("");
    const dW = line(Yw, "wWin"), dM = line(Ym, "wMargin");
    const x0 = X(pts[0].i), x1 = X(pts[pts.length - 1].i);
    el("path", { d: dW + `L${x1} ${Yw(0.5)}L${x0} ${Yw(0.5)}Z`, fill: "var(--w-acc)", opacity: 0.1 }, svg);
    el("path", { d: dW, fill: "none", stroke: "var(--w-acc)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
    el("path", { d: dM + `L${x1} ${Ym(0)}L${x0} ${Ym(0)}Z`, fill: "var(--margin)", opacity: 0.1 }, svg);
    el("path", { d: dM, fill: "none", stroke: "var(--margin)", "stroke-width": 2, "stroke-linejoin": "round" }, svg);
  }
  // current position cursor + markers
  el("line", { x1: X(ply), x2: X(ply), y1: 2, y2: top2 + CH.h2 + 2, stroke: "var(--b-acc)", "stroke-width": 2 }, svg);
  const ev = hasEv(cur.moves[ply]) ? cur.moves[ply].ev : null;
  if (ev) {
    el("circle", { cx: X(ply), cy: Yw(ev.wWin), r: 4, fill: "var(--w-acc)", stroke: "#fff", "stroke-width": 2 }, svg);
    el("circle", { cx: X(ply), cy: Ym(ev.wMargin), r: 4, fill: "var(--margin)", stroke: "#fff", "stroke-width": 2 }, svg);
  }
  // hover crosshair layer
  const hover = el("line", { y1: 2, y2: top2 + CH.h2 + 2, stroke: "#9aa5b1", "stroke-width": 1, visibility: "hidden" }, svg);
  svg._hover = hover;
}

function chartPlyAt(e) {
  const { svg, w, n } = chartGeom();
  const r = svg.getBoundingClientRect();
  const x = (e.clientX - r.left) * (w / r.width);
  return Math.max(0, Math.min(n, Math.round(((x - CH.l) / (w - CH.l - CH.r)) * n)));
}

// Scroll `row` into view inside its own list only: element.scrollIntoView would also scroll the page (single-column layout).
function scrollWithin(box, row) {
  const b = box.getBoundingClientRect(), r = row.getBoundingClientRect();
  if (r.top < b.top) box.scrollTop -= b.top - r.top;
  else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom;
}

// ---- Main update -----------------------------------------------------------
function update(scrollList = true) {
  if (!cur) return;
  ply = Math.max(0, Math.min(ply, cur.moves.length));
  const st = stateAt(cur, ply);
  const paths = { B: shortestPaths(st.block, st.B, "B"), W: shortestPaths(st.block, st.W, "W") };
  drawBoard(st, paths);
  renderPathTile(st, paths);
  const h = games[curGameIdx];
  drawWallsCounter("wallsB", st.used.B, h.wb);
  drawWallsCounter("wallsW", st.used.W, h.ww);
  $("subB").textContent = `→ row 1 · ${h.wb - st.used.B} walls`;
  $("subW").textContent = `→ row 9 · ${h.ww - st.used.W} walls`;
  for (const [id, name] of [["modelB", cur.props.PB?.[0]], ["modelW", cur.props.PW?.[0]]])
    $(id).innerHTML = match && name ? `${mDot(name)}${mLabel(name)} · ${escapeHtml(shortName(name))}` : "";
  const over = ply === cur.moves.length;
  $("cardB").classList.toggle("tomove", !over && st.toMove === "B");
  $("cardW").classList.toggle("tomove", !over && st.toMove === "W");
  const last = ply > 0 ? cur.moves[ply - 1] : null;
  $("moveInfo").textContent = over ? `End · ${resultLabel(games[curGameIdx])}` : `${ply} / ${cur.moves.length}` + (last ? ` · ${last.pla} ${notation(last)}` : "");
  $("moveSlider").max = cur.moves.length;
  $("moveSlider").value = ply;
  renderEvalStrip();
  document.querySelectorAll("#moveList .m.cur").forEach(e => e.classList.remove("cur"));
  const row = document.querySelector(`#moveList .m[data-i="${ply}"]`);
  if (row) { row.classList.add("cur"); if (scrollList) scrollWithin($("moveList"), row); }
  renderChart();
}

// ---- Game list -------------------------------------------------------------
// Sort keys for finding interesting games. desc = true means "most interesting first" is descending.
// Winner-based metrics are null for draws (no winner), so draws sort last on them.
const SORTS = [
  { key: "idx", label: "File order", desc: "Games in the order they appear in the file.", get: s => s.i, fmt: v => v + 1, dir: "asc" },
  { key: "comeback", label: "Comeback (win%)", desc: "Winner's lowest White/Black win% during search. Low = the winner was nearly lost. Draws have no winner and are listed last.", get: s => s.minWinnerP, fmt: v => pct(v, 0), dir: "asc" },
  { key: "scoreComeback", label: "Comeback (lead)", desc: "Winner's worst predicted lead during search. Negative = the winner was predicted to trail by that much. Draws are listed last.", get: s => leadOnly(s) ? s.minWinnerScore : null, fmt: fmtScore, dir: "asc" },
  { key: "raceDeficit", label: "Race comeback (path)", desc: "Winner's worst path-race deficit: max over positions of (winner's shortest path − loser's). High = the winner was far behind in the race. Draws are listed last.", get: s => s.raceDeficit, fmt: v => (v > 0 ? "+" : "") + v, dir: "desc" },
  { key: "leadChanges", label: "Lead changes", desc: "Times the win% crossed 50% during search.", get: s => s.leadChanges, fmt: v => v, dir: "desc" },
  { key: "volatility", label: "Volatility", desc: "Total win% movement: sum of |Δ win%| between consecutive searches.", get: s => s.volatility, fmt: v => (100 * v).toFixed(0), dir: "desc" },
  { key: "maxSwing", label: "Biggest swing", desc: "Largest single-move |Δ win%|: a blunder or a surprise.", get: s => s.maxSwing, fmt: v => pct(v, 0), dir: "desc" },
  { key: "decidedLeft", label: "Decided late", desc: "Moves left after the winner last dropped below 90%. Low = the game stayed open until the end. Draws are listed last.", get: s => s.decidedLeft, fmt: v => v, dir: "asc" },
  { key: "close", label: "Close finish", desc: "Final result margin. Small = close game. Draws and resignations / forfeits are listed last.", get: s => s.margin, fmt: v => v, dir: "asc" },
  { key: "cycling", label: "Cycling", desc: "Longest repeated stretch: plies in a row whose position (pawns, walls placed, side to move) occurred before in the game. High = the game cycled.", get: s => s.cycleMax, fmt: v => v, dir: "desc" },
  { key: "komi", label: "Komi", desc: "Komi (KM, standard −0.5; positive favours White).", get: s => s.komi, fmt: fmtKomi, dir: "asc" },
  { key: "maxPath", label: "Longest detour", desc: "Longest shortest path of either pawn at any point (a clear 9x9 board is 8). High = a big maze.", get: s => s.maxPath, fmt: v => v, dir: "desc" },
  { key: "walls", label: "Walls placed", desc: "Total walls placed by both sides.", get: s => s.walls, fmt: v => v, dir: "desc" },
  { key: "brier", label: "Misjudged", desc: "Mean squared error of White win% vs the actual result (Brier, a draw counts as ½). High = the search was wrong for long stretches.", get: s => s.brier, fmt: v => v.toFixed(2), dir: "desc" },
  { key: "len", label: "Length", desc: "Total moves.", get: s => s.n, fmt: v => v, dir: "desc" },
];
let sortDesc = false;
// Lead-based metrics use only games that carry lead=; if none do (old files) the old score margin is used instead.
const leadOnly = s => s.hasLead || !statsCache.some(t => t.hasLead);
function sortNote(M) {
  if (M.key !== "scoreComeback") return M.desc;
  const nLead = statsCache.filter(t => t.hasLead).length, nEv = statsCache.filter(t => t.minWinnerScore !== null).length;
  if (!nLead) return "Margin (no lead in file): " + M.desc.replace("predicted lead", "predicted margin (old score)").replace("trail", "lose");
  return M.desc + (nLead < nEv ? ` ${nEv - nLead} games without lead= skipped.` : "");
}
function initSortSelect() {
  $("sortKey").innerHTML = SORTS.map(s => `<option value="${s.key}">${s.label}</option>`).join("");
}
function setSort(key) {
  const M = SORTS.find(s => s.key === key) || SORTS[0];
  $("sortKey").value = M.key;
  sortDesc = M.dir === "desc";
  applyFilters();
  $("gameList").scrollTop = 0;
}
function randomGame() { if (filtered.length) selectGame(filtered[Math.floor(Math.random() * filtered.length)]); }

// ---- Filters: one shared state for the game list and the statistics page ----
// The selects hold the state; `openingFilter` (set from the statistics page's opening table) is shown as a chip.
const FILTER_IDS = ["fRes", "fRule", "fKomi", "fWalls", "fType", "fCyc", "fMin"];
let openingFilter = "";
let repTags = false;   // tag each list row with its repetition rule (the file mixes rules)
function passes(g, s) {
  const r = $("fRes").value, rule = $("fRule").value, km = $("fKomi").value, wl = $("fWalls").value;
  const t = $("fType").value, cyc = +$("fCyc").value || 0, mn = +$("fMin").value || 0;
  if (r) {
    if (r[0] === "m") { if (!match || g.winnerModel !== match.models[+r[1]]) return false; }
    else if (r[0] === "a") { if (!match || g.pb !== match.models[+r[1]]) return false; }
    else if (r === "D") { if (!isDraw(g.winner)) return false; }
    else if (r.startsWith("D:")) { if (!isDraw(g.winner) || g.drawReason !== r.slice(2)) return false; }
    else if (g.winner !== r) return false;
  }
  if (rule && !(rule === "on" ? g.repetition > 0 : rule === "off" ? g.ruleKnown && !g.repetition : g.repetition === +rule)) return false;
  if (km && g.komi !== (km === "std" ? STANDARD_KOMI : +km)) return false;
  if (wl) {
    const std = g.wb === STANDARD_WALLS && g.ww === STANDARD_WALLS;
    if (wl === "std" ? !std : wl === "hc" ? std : wl === "hcB" ? !(g.wb < g.ww) : !(g.ww < g.wb)) return false;
  }
  if (t && g.gtype !== t) return false;
  if (g.nMoves < mn) return false;
  if (openingFilter && g.opening !== openingFilter && !g.opening.startsWith(openingFilter + " ")) return false;
  if (cyc && s && s.cycleMax < cyc) return false;   // before the summaries are ready the cycling filter is not applied
  return true;
}
// Human-readable list of the active filters ("" when none), for the statistics page.
function activeFilters() {
  const out = [];
  for (const id of FILTER_IDS) {
    const e = $(id);
    if (!e.value || e.hidden) continue;
    out.push(id === "fMin" ? `≥ ${e.value} moves` : e.options[e.selectedIndex].text);
  }
  if (openingFilter) out.push(`opening ${openingLabel(openingFilter)}`);
  return out;
}
function setOpeningFilter(key) { openingFilter = key; applyFilters(); }
function clearFilters() {
  for (const id of FILTER_IDS) $(id).value = "";
  openingFilter = "";
  applyFilters();
}
function renderChips() {
  const n = activeFilters().length;
  $("fChips").innerHTML = (openingFilter ? `<span class="chip">Opening ${escapeHtml(openingLabel(openingFilter))} <button data-x="opening" title="Remove">×</button></span>` : "") +
    (n ? `<button class="linkbtn" data-x="all">Clear filters</button>` : "");
}

// Fill the filter selects from the games of a newly loaded file; a select whose filter cannot split the file is hidden.
function initFilters() {
  const res = `<option value="">All results</option><option value="B">Black wins</option><option value="W">White wins</option>` +
    (games.some(g => isDraw(g.winner)) ? `<option value="D">Draws</option>` +
      [...new Set(games.filter(g => isDraw(g.winner) && g.drawReason).map(g => g.drawReason))].sort()
        .map(r => `<option value="D:${escapeHtml(r)}">Draw by ${DRAW_REASONS[r] || escapeHtml(r)}</option>`).join("") : "");
  $("fRes").innerHTML = res + (match ? match.labels.map((l, k) => `<option value="m${k}">${l} wins</option>`).join("") +
    match.labels.map((l, k) => `<option value="a${k}">${l} as Black</option>`).join("") : "");
  const reps = new Set(games.map(g => g.ruleKnown ? g.repetition : -1));
  repTags = reps.size > 1;
  $("fRule").hidden = !repTags;
  $("fRule").innerHTML = `<option value="">Any rep. rule</option><option value="on">Rep. rule on</option><option value="off">Rep. rule off</option>` +
    [...reps].filter(x => x > 0).sort().map(x => `<option value="${x}">Rep. rule N = ${x}</option>`).join("");
  const komis = [...new Set(games.map(g => g.komi))].sort((a, b) => a - b);
  $("fKomi").hidden = komis.length < 2;
  $("fKomi").innerHTML = `<option value="">Any komi</option><option value="std">Komi standard (−0.5)</option>` +
    komis.filter(k => k !== STANDARD_KOMI).map(k => `<option value="${k}">Komi ${fmtKomi(k)}</option>`).join("");
  $("fWalls").hidden = games.every(g => g.wb === STANDARD_WALLS && g.ww === STANDARD_WALLS);
  const types = [...new Set(games.map(g => g.gtype).filter(Boolean))].sort();
  $("fType").hidden = types.length < 2;
  $("fType").innerHTML = `<option value="">All types</option>` + types.map(t => `<option>${escapeHtml(t)}</option>`).join("");
  for (const id of FILTER_IDS) $(id).value = "";
  openingFilter = "";
}

function applyFilters() {
  filtered = [];
  for (const g of games) if (passes(g, statsCache && statsCache[g.i])) filtered.push(g.i);
  const M = SORTS.find(s => s.key === $("sortKey").value) || SORTS[0];
  const sign = (sortDesc ? -1 : 1);
  if (statsCache) {
    const val = i => M.get(statsCache[i]);
    filtered.sort((a, b) => {
      const va = val(a), vb = val(b);
      if (va === null || vb === null) return (va === null) - (vb === null) || a - b;   // missing values last
      return sign * (va - vb) || a - b;
    });
  }
  if (statsCache) $("sortKey").querySelector('option[value="scoreComeback"]').textContent = statsCache.some(t => t.hasLead) || !statsCache.length ? "Comeback (lead)" : "Comeback (margin, no lead)";
  $("sortDesc").textContent = statsCache ? sortNote(M) : "Computing game metrics…";
  $("sortDir").textContent = sortDesc ? "↓" : "↑";
  const tag = s => `<span class="tag">${escapeHtml(s)}</span>`;
  $("gameList").innerHTML = filtered.map(i => {
    const g = games[i];
    let tags = g.gtype && g.gtype !== "normal" ? tag(g.gtype) : "";
    if (repTags) tags += tag(repLabel(g) || "rule ?");
    if (g.rulesLabel) tags += tag(g.rulesLabel);
    if (g.drawReason) tags += tag(DRAW_REASONS[g.drawReason] || g.drawReason);
    if (match && g.winnerModel) tags += `<span class="won">${mDot(g.winnerModel)}${mLabel(g.winnerModel)} won</span>`;
    const v = statsCache && M.key !== "idx" && M.key !== "len" ? M.get(statsCache[i]) : undefined;
    const mval = v === undefined ? "" : v === null ? "—" : M.fmt(v);
    return `<div class="g${i === curGameIdx ? " sel" : ""}" data-g="${i}"><span class="idx">#${i + 1}</span><span class="res ${isDraw(g.winner) ? "D" : g.winner}">${escapeHtml(g.result)}</span><span class="meta">${g.nMoves} moves${tags}</span><span class="mval">${mval}</span></div>`;
  }).join("");
  $("listCount").textContent = `${filtered.length} / ${games.length} games` + (activeFilters().length ? " (filtered)" : "");
  $("gamesHeadInfo").textContent = curGameIdx >= 0 ? `· #${curGameIdx + 1} of ${games.length}` : `· ${games.length}`;
  renderChips();
  if (statsCache && document.body.classList.contains("stats")) renderStats();
}

function selectGame(i) {
  if (i < 0 || i >= games.length) return;
  curGameIdx = i;
  cur = parseGame(lines[i]);
  ply = cur.startTurnIdx;
  document.querySelectorAll("#gameList .g.sel").forEach(e => e.classList.remove("sel"));
  const row = document.querySelector(`#gameList .g[data-g="${i}"]`);
  if (row) { row.classList.add("sel"); scrollWithin($("gameList"), row); }
  $("gamesHeadInfo").textContent = `· #${i + 1} of ${games.length}`;
  renderInfo();
  renderMoveList();
  update();
}

function stepGame(d) {
  if (!filtered.length) return;
  let pos = filtered.indexOf(curGameIdx);
  pos = pos < 0 ? 0 : Math.max(0, Math.min(filtered.length - 1, pos + d));
  selectGame(filtered[pos]);
}

// ---- Loading ---------------------------------------------------------------
function loadText(text, name) {
  lines = text.split(/\r?\n/).filter(l => l.startsWith("(;"));
  games = lines.map(headerOf);
  match = detectMatch();
  statsCache = null;
  initFilters();
  $("fileName").textContent = `${name} — ${games.length} games` + (match ? ` · match: ${match.models.map((m, k) => `${match.labels[k]} ${shortName(m)}`).join(" vs ")}` : " · self-play");
  document.querySelectorAll(".vis").forEach(e => e.hidden = false);
  $("empty").hidden = true;
  curGameIdx = -1;
  applyFilters();
  $("gamesPanel").classList.remove("open");
  if (games.length) selectGame(0);
  computeSummaries();
}

// Per-game summaries (sort metrics + statistics) are computed in chunks so the page stays responsive.
let summaryJob = 0;
function computeSummaries() {
  const job = ++summaryJob, out = new Array(lines.length);
  let i = 0;
  const step = () => {
    if (job !== summaryJob) return;              // a newer file was loaded
    const end = Math.min(lines.length, i + 400);
    for (; i < end; i++) out[i] = summarize(lines[i], games[i]);
    if (i < lines.length) { $("sortDesc").textContent = `Computing game metrics… ${Math.round(100 * i / lines.length)}%`; return setTimeout(step, 0); }
    statsCache = out;
    applyFilters();   // re-renders the statistics page when it is open
  };
  step();
}

function loadFile(file) {
  const r = new FileReader();
  r.onload = () => loadText(r.result, file.name);
  r.readAsText(file);
}

