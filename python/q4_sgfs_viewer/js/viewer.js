"use strict";
// ---- DOM / SVG helpers -------------------------------------------------------
const $ = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const SEAT_COL = ["var(--s-acc)", "var(--w-acc)", "var(--n-acc)", "var(--e-acc)"];
const SEAT_HEX = ["#0072b2", "#e69f00", "#cc79a7", "#009e73"];
const DRAW_COL = "var(--draw)";
function el(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, attrs = {}) {
  const e = el("text", { x, y, "font-size": 10.5, fill: "#8a96a3", ...attrs }, parent);
  e.textContent = s;
  return e;
}
const escapeHtml = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";
const tip = $("tip");
function showTip(e, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 6) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 6) y = e.clientY - r.height - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.display = "none"; }
const sw = c => `<span class="sw" style="background:${c}"></span>`;

// scrolls a list container (never the page, which on a phone would jump away from the board) to show a row
function scrollWithin(box, row) {
  if (!row) return;
  const b = box.getBoundingClientRect(), r = row.getBoundingClientRect();
  if (r.top < b.top) box.scrollTop -= b.top - r.top;
  else if (r.bottom > b.bottom) box.scrollTop += r.bottom - b.bottom;
}

// ---- state -------------------------------------------------------------------
let lines = [], heads = [], sums = [], listOrder = [];
let fileLabel = "", fullVisits = null, sumDone = 0;
let curIdx = -1, cur = null, ply = 0, rot = 0, showPaths = true;
let sortKey = "file", sortDesc = false;
let loadToken = 0;

// ---- names -------------------------------------------------------------------
function modelOf(h, i) {
  const n = h.nets[i];
  if (!n) return h.types[i] === "selfplay" ? h.names[i] : "";
  const parts = n.split("/").filter(Boolean);
  return /^model(\.bin)?(\.gz)?$/.test(parts[parts.length - 1]) && parts.length > 1 ? parts[parts.length - 2] : parts[parts.length - 1];
}
function seatShort(h, i) { return h.types[i] || "?"; }
function playersTag(h) {
  const t = h.types.map((_, i) => seatShort(h, i) + "|" + modelOf(h, i));
  if (t.every(x => x === t[0])) return `4×${seatShort(h, 0)}`;
  return [0, 1, 2, 3].map(i => SEATS[i] + ":" + seatShort(h, i)).join(" ");
}
const ruleLabel = h => h.rep > 0 ? "rep " + h.rep : "rep off";
const wallsStd = h => h.walls.every(w => w === 7);
function resultLabel(h) {
  if (h.winner >= 0) return SEATS[h.winner];
  if (h.isDraw) return "D";
  return "?";
}
function resultFull(h) {
  if (h.winner >= 0) return SEAT_NAMES[h.winner] + " (" + SEATS[h.winner] + ") wins";
  if (h.isDraw) return "Draw" + (h.dr ? " by " + h.dr : "");
  return "Unfinished";
}

// ---- loading -------------------------------------------------------------------
function loadFile(file) {
  const r = new FileReader();
  r.onload = () => loadText(r.result, file.name);
  r.readAsText(file);
}
function loadText(text, name) {
  const token = ++loadToken;
  fileLabel = name;
  const all = text.split(/\r?\n/).filter(l => l.startsWith("(;"));
  lines = []; heads = []; sums = []; listOrder = []; curIdx = -1; cur = null; sumDone = 0; fullVisits = null;
  const errors = [];
  let first = null;
  for (const l of all) {
    try { const h = headerOf(l); lines.push(l); heads.push(h); sums.push(null); }
    catch (e) { if (first === null) first = e.message; errors.push(e.message); }
  }
  $("fileName").textContent = name;
  if (!lines.length) {
    $("fileName").textContent = `${name}: ${first || "no games"}`;
    $("empty").hidden = false;
    for (const e of document.querySelectorAll(".vis")) e.hidden = true;
    renderList();
    return;
  }
  if (errors.length) $("fileName").textContent = `${name} (${errors.length} lines skipped: ${first})`;
  initFilterOptions();
  applyFilters();
  selectGame(listOrder.length ? listOrder[0] : 0);
  runSummaryPass(token);
}
// background pass: parse each game fully once for its summary (swing, early decision, calibration points, ...)
function runSummaryPass(token) {
  const step = () => {
    if (token !== loadToken) return;
    const t0 = performance.now();
    while (sumDone < lines.length && performance.now() - t0 < 30) {
      try { sums[sumDone] = summarize(parseGame(lines[sumDone])); } catch (e) { sums[sumDone] = summarize({ h: heads[sumDone], events: [] }); }
      sumDone++;
    }
    $("gamesHeadInfo").textContent = sumDone < lines.length ? `· summaries ${sumDone}/${lines.length}` : "";
    if (sumDone < lines.length) setTimeout(step, 0);
    else {
      computeFullVisits();
      if (sortKey === "swing" || sortKey === "early") applyFilters(true);
      if (document.body.classList.contains("stats")) renderStats();
    }
  };
  setTimeout(step, 0);
}
function computeFullVisits() {
  const v = [];
  for (const s of sums) if (s) for (let i = 0; i < s.visits.length; i++) if (s.weights[i] > 0) v.push(s.visits[i]);
  if (!v.length) for (const s of sums) if (s) for (const x of s.visits) v.push(x);
  v.sort((a, b) => a - b);
  fullVisits = v.length ? v[v.length >> 1] : null;
}

// ---- filters ---------------------------------------------------------------------
const FILTER_IDS = ["fRes", "fRule", "fWalls", "fType", "fTable", "fPType", "fModel", "fElim", "fMin", "fMax"];
const RES_OPTS = [["", "All results"], ["S", "South wins"], ["W", "West wins"], ["N", "North wins"], ["E", "East wins"],
  ["draw", "Any draw"], ["draw-rep", "Draw by repetition"], ["draw-max", "Draw by maxPlies"], ["unfinished", "Unfinished"]];
function setOptions(sel, first, values) {
  const old = sel.value;
  sel.innerHTML = "";
  for (const [v, t] of first.concat(values.map(x => [x, x]))) { const o = document.createElement("option"); o.value = v; o.textContent = t; sel.appendChild(o); }
  sel.value = [...sel.options].some(o => o.value === old) ? old : "";
}
function uniq(f) { const s = new Set(); for (const h of heads) for (const x of f(h)) if (x) s.add(x); return [...s].sort(); }
function initFilterOptions() {
  setOptions($("fRes"), [["", "All results"]].concat(RES_OPTS.slice(1)), []);
  setOptions($("fRule"), [["", "Any rule"], ["on", "Repetition on"], ["off", "Repetition off"]], uniq(h => h.rep > 0 ? ["rep=" + h.rep] : []));
  setOptions($("fType"), [["", "All types"]], uniq(h => [h.gtype]));
  const tables = uniq(h => [h.table]);
  setOptions($("fTable"), [["", "All tables"]], tables);
  $("fTable").hidden = !tables.length;
  setOptions($("fPType"), [["", "Any player type"]], uniq(h => h.types));
  setOptions($("fModel"), [["", "Any model"]], uniq(h => [0, 1, 2, 3].map(i => modelOf(h, i))));
}
function passes(h) {
  const f = k => $(k).value;
  const rk = resultKey(h), r = f("fRes");
  if (r === "draw" ? !h.isDraw : r && rk !== r) return false;
  const rule = f("fRule");
  if (rule === "on" ? h.rep <= 0 : rule === "off" ? h.rep > 0 : rule && "rep=" + h.rep !== rule) return false;
  const w = f("fWalls");
  if (w === "std" ? !wallsStd(h) : w === "other" && wallsStd(h)) return false;
  if (f("fType") && h.gtype !== f("fType")) return false;
  if (f("fTable") && h.table !== f("fTable")) return false;
  if (f("fPType") && !h.types.includes(f("fPType"))) return false;
  if (f("fModel") && ![0, 1, 2, 3].some(i => modelOf(h, i) === f("fModel"))) return false;
  if (f("fElim") === "yes" ? h.elims === 0 : f("fElim") === "no" && h.elims > 0) return false;
  if (f("fMin") !== "" && h.plies < +f("fMin")) return false;
  if (f("fMax") !== "" && h.plies > +f("fMax")) return false;
  return true;
}
function activeFilters() {
  const out = [];
  for (const id of FILTER_IDS) {
    const e = $(id);
    if (e.value === "") continue;
    out.push(e.tagName === "SELECT" ? e.options[e.selectedIndex].textContent : (id === "fMin" ? "plies ≥ " : "plies ≤ ") + e.value);
  }
  return out;
}
function clearFilters() { for (const id of FILTER_IDS) $(id).value = ""; applyFilters(); }

// ---- sorting ----------------------------------------------------------------------
const SORTS = {
  file: ["File order", "games in the order of the file", i => i],
  plies: ["Plies", "number of moves (eliminations do not count)", i => heads[i].plies],
  winner: ["Winner", "S, W, N, E winners, then draws, then unfinished", i => heads[i].winner >= 0 ? heads[i].winner : heads[i].isDraw ? 4 : 5],
  swing: ["Largest swing", "the largest change of any seat's win probability within the game (needs the summary pass)", i => sums[i] && sums[i].hasEval ? sums[i].swing : null],
  early: ["Decided early", "the earliest ply where some seat's win probability exceeds 0.9 (earlier first; needs the summary pass)", i => sums[i] && sums[i].early !== null ? sums[i].early : null],
  draw: ["Draws first", "draws first, then unfinished, then decided games", i => heads[i].isDraw ? 0 : heads[i].finished ? 2 : 1],
};
function initSortSelect() {
  const sel = $("sortKey");
  for (const k in SORTS) { const o = document.createElement("option"); o.value = k; o.textContent = SORTS[k][0]; sel.appendChild(o); }
  const o = document.createElement("option"); o.value = "plies_rev"; o.textContent = "Plies (longest first)"; sel.insertBefore(o, sel.options[2]);
}
function setSort(k) { sortKey = k; sortDesc = k === "swing"; applyFilters(); }
function applyFilters(keepCur) {
  const keep = curIdx;
  let idx = [];
  for (let i = 0; i < heads.length; i++) if (passes(heads[i])) idx.push(i);
  const rev = sortKey === "plies_rev", key = rev ? "plies" : sortKey, f = SORTS[key][2];
  idx.sort((a, b) => {
    const x = f(a), y = f(b);
    if (x === null && y === null) return a - b;
    if (x === null) return 1;
    if (y === null) return -1;
    return ((x - y) * (rev ? -1 : 1) * (sortDesc ? -1 : 1)) || a - b;
  });
  listOrder = idx;
  $("sortDesc").textContent = SORTS[key][1] + (rev ? ", longest first" : "");
  $("sortDir").textContent = sortDesc ? "↑" : "↓";
  renderList();
  if (document.body.classList.contains("stats")) renderStats();
  if (!keepCur && keep >= 0 && !listOrder.includes(keep) && listOrder.length) selectGame(listOrder[0]);
}

// ---- game list ---------------------------------------------------------------------
const LIST_MAX = 400;
function renderList() {
  const rows = [];
  const uniform = f => heads.length > 0 && heads.every(h => f(h) === f(heads[0]));
  const showRule = !uniform(h => h.rep);
  for (const i of listOrder.slice(0, LIST_MAX)) {
    const h = heads[i], tags = [];
    tags.push(`<span class="tag">${escapeHtml(playersTag(h))}</span>`);
    if (showRule) tags.push(`<span class="tag t-rule">${ruleLabel(h)}</span>`);
    if (!wallsStd(h)) tags.push(`<span class="tag">walls ${h.walls.join("/")}</span>`);
    if (h.gtype !== "normal") tags.push(`<span class="tag t-gt">${escapeHtml(h.gtype)}</span>`);
    if (h.table) tags.push(`<span class="tag t-match">${escapeHtml(h.table)} #${h.opening}.${h.rotation}</span>`);
    if (h.elims) tags.push(`<span class="tag t-elim">elim ${h.elims}</span>`);
    if (h.isDraw && h.dr) tags.push(`<span class="tag">${escapeHtml(h.dr)}</span>`);
    const cls = h.winner >= 0 ? SEATS[h.winner] : h.isDraw ? "D" : "U";
    rows.push(`<div class="g${i === curIdx ? " sel" : ""}" data-g="${i}"><span class="idx">${i + 1}</span><span class="res ${cls}">${resultLabel(h)}</span><span class="meta">${tags.join("")}</span><span class="mval">${h.plies}</span></div>`);
  }
  $("gameList").innerHTML = rows.join("");
  const f = activeFilters();
  $("fChips").innerHTML = f.length ? f.map(t => `<span class="chip">${escapeHtml(t)}</span>`).join("") + `<button class="linkbtn" data-x="all">clear</button>` : "";
  $("listCount").textContent = heads.length ? `${listOrder.length} of ${heads.length} games` + (listOrder.length > LIST_MAX ? ` (first ${LIST_MAX} shown)` : "") : "";
}
function selectGame(i) {
  if (i < 0 || i >= lines.length) return;
  curIdx = i;
  try { cur = parseGame(lines[i]); } catch (e) { cur = null; $("fileName").textContent = `${fileLabel}: game ${i + 1}: ${e.message}`; return; }
  ply = cur.events.length;
  for (const e of document.querySelectorAll(".vis")) e.hidden = false;
  $("empty").hidden = true;
  $("moveSlider").max = cur.events.length;
  for (const r of document.querySelectorAll("#gameList .g")) r.classList.toggle("sel", +r.dataset.g === i);
  scrollWithin($("gameList"), document.querySelector("#gameList .g.sel"));
  curBlunder = blunderOf(cur);
  renderInfo();
  renderMoveList();
  update();
}
let curBlunder = null;
function stepGame(d) {
  const p = listOrder.indexOf(curIdx);
  const q = p < 0 ? 0 : Math.max(0, Math.min(listOrder.length - 1, p + d));
  if (listOrder.length) selectGame(listOrder[q]);
}
function randomGame() { if (listOrder.length) selectGame(listOrder[Math.floor(Math.random() * listOrder.length)]); }

// ---- board ----------------------------------------------------------------------------
const CELL = 40, GAP = 6, PAD = 30, STEP = CELL + GAP, BSIZE = PAD * 2 + 11 * CELL + 10 * GAP;
// n quarter turns counter-clockwise: south -> west -> north -> east at the bottom
function rotCell(x, y, n) { for (let k = 0; k < n; k++) { const t = x; x = 10 - y; y = t; } return [x, y]; }
const cx = c => PAD + c * STEP;               // display cell column -> left pixel
const cy = r => PAD + (10 - r) * STEP;        // display cell row (up) -> top pixel
function cellName(x, y) { return String.fromCharCode(97 + x) + (y + 1); }
function renderBoard() {
  const svg = $("board");
  svg.setAttribute("viewBox", `0 0 ${BSIZE} ${BSIZE}`);
  svg.innerHTML = "";
  el("rect", { x: 0, y: 0, width: BSIZE, height: BSIZE, rx: 14, fill: "var(--board)" }, svg);
  const st = cur.states[ply], dpaths = pathsAt(cur, ply), paths = showPaths ? dpaths : null;
  const n = rot % 4;
  const last = st.last;
  for (let y = 0; y < 11; y++) for (let x = 0; x < 11; x++) {
    const [dx, dy] = rotCell(x, y, n), px = cx(dx), py = cy(dy);
    const goal = x === 5 && y === 5;
    el("rect", { x: px, y: py, width: CELL, height: CELL, rx: 5, fill: goal ? "#f6e7a8" : "var(--cell)" }, svg);
    if (goal) { txt(svg, px + CELL / 2, py + CELL / 2 + 5, "★", { "text-anchor": "middle", "font-size": 18, fill: "#b8860b" }); }
    if (paths) {
      for (let s = 0; s < 4; s++) {
        const set = paths.cells[s];
        if (set && set.has(x + y * 11)) {
          const qx = px + (s % 2 ? CELL / 2 : 0), qy = py + (s < 2 ? 0 : CELL / 2);
          el("rect", { x: qx + 2, y: qy + 2, width: CELL / 2 - 4, height: CELL / 2 - 4, rx: 3, fill: SEAT_HEX[s], opacity: 0.8 }, svg);
        }
      }
    }
  }
  // coordinate labels: the logical name of the cell on the bottom and left edge
  for (let d = 0; d < 11; d++) {
    // bottom edge: display column d, row 0 ; left edge: display column 0, row d
    const inv = (dc, dr) => { let c = [dc, dr]; for (let k = 0; k < (4 - n) % 4; k++) c = [10 - c[1], c[0]]; return c; };
    const [bx, by] = inv(d, 0), [lx, ly] = inv(0, d);
    txt(svg, cx(d) + CELL / 2, BSIZE - 10, cellName(bx, by), { "text-anchor": "middle", "font-size": 9.5, fill: "#a9bcc6" });
    txt(svg, 6, cy(d) + CELL / 2 + 3, cellName(lx, ly), { "font-size": 9.5, fill: "#a9bcc6" });
  }
  // walls
  const drawWall = (w, extra) => {
    const cells = [[w.x, w.y], [w.x + 1, w.y], [w.x, w.y + 1], [w.x + 1, w.y + 1]].map(([a, b]) => rotCell(a, b, n));
    const c0 = Math.min(...cells.map(c => c[0])), r0 = Math.min(...cells.map(c => c[1]));
    const x0 = cx(c0), x1 = cx(c0 + 1) + CELL, yt = cy(r0 + 1), yb = cy(r0) + CELL, mx = (x0 + x1) / 2, my = (yt + yb) / 2;
    const horiz = w.h !== (n % 2 === 1);
    const a = horiz ? [x0, my, x1, my] : [mx, yt, mx, yb];
    if (extra) el("line", { x1: a[0], y1: a[1], x2: a[2], y2: a[3], stroke: "#fff", "stroke-width": GAP + 5, "stroke-linecap": "round" }, svg);
    el("line", { x1: a[0], y1: a[1], x2: a[2], y2: a[3], stroke: SEAT_HEX[w.seat], "stroke-width": GAP + 1, "stroke-linecap": "butt" }, svg);
  };
  const lastWall = last && !last.elim && last.action && last.action.wall ? cur.walls[st.nw - 1] : null;
  for (let i = 0; i < st.nw; i++) if (cur.walls[i] !== lastWall) drawWall(cur.walls[i], false);
  if (lastWall) drawWall(lastWall, true);
  // pawns
  for (let s = 0; s < 4; s++) {
    const p = st.pawns[s];
    if (!p) continue;
    const [dx, dy] = rotCell(p[0], p[1], n), px = cx(dx) + CELL / 2, py = cy(dy) + CELL / 2;
    const isLast = last && !last.elim && last.seat === s && last.action && !last.action.wall;
    if (isLast) el("circle", { cx: px, cy: py, r: 19.5, fill: "none", stroke: "#fff", "stroke-width": 3 }, svg);
    el("circle", { cx: px, cy: py, r: 15, fill: SEAT_HEX[s], stroke: "#fff", "stroke-width": 2 }, svg);
    txt(svg, px, py + 5, SEATS[s], { "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#fff" });
  }
  return dpaths;
}

// ---- panels -----------------------------------------------------------------------------
function renderCards(dist) {
  const st = cur.states[ply], h = cur.h, mover = moverAt(cur, ply);
  let html = "";
  for (let s = 0; s < 4; s++) {
    const alive = st.alive[s];
    const total = h.walls[s], left = st.left[s];
    let pips = "";
    for (let k = 0; k < total; k++) pips += `<i class="${k < left ? "" : "used"}"></i>`;
    const model = modelOf(h, s);
    const sub = !alive ? "eliminated" : (dist[s] === null ? "" : (dist[s] < 0 ? "no path" : `${dist[s]} steps`)) + (model && model !== h.names[s] ? " · " + model : "");
    html += `<div class="pcard${s === mover && alive ? " tomove" : ""}${alive ? "" : " out"}" style="--acc:${SEAT_COL[s]}"><div class="seatdot">${SEATS[s]}</div><div class="nm"><div class="name">${escapeHtml(h.names[s])}</div><div class="sub">${escapeHtml(h.types[s] || "?")}${h.visits[s] ? " @" + h.visits[s] : ""} · ${escapeHtml(sub)}</div></div><div class="walls">${pips}</div></div>`;
  }
  $("players").innerHTML = html;
}
function renderEval(dist) {
  const ev = ply < cur.events.length ? cur.events[ply] : null, c = ev && ev.comment;
  const bar = $("evBar"), nums = $("evNums");
  if (c) {
    const colors = SEAT_COL.concat([DRAW_COL]), names = ["S", "W", "N", "E", "draw"];
    bar.innerHTML = c.p.map((p, i) => `<div style="flex:${Math.max(p, 0.0001)};background:${colors[i]}"></div>`).join("");
    nums.innerHTML = c.p.map((p, i) => `<span style="--c:${colors[i]}">${names[i]} <b>${pct(p)}</b></span>`).join("");
    $("evVisV").textContent = c.visits;
    $("evVisS").textContent = `weight ${c.weight.toFixed(2)}` + (fullVisits !== null && c.visits < fullVisits ? " · cheap search" : "");
    $("evWinK").textContent = `Win probability · search before ${SEATS[ev.seat]}'s move`;
  } else {
    bar.innerHTML = ""; nums.innerHTML = `<span class="note">${ev ? (ply < cur.h.startTurnIdx ? "opening ply, not played by the engines" : "no search for this move") : "end of the game"}</span>`;
    $("evVisV").textContent = "—"; $("evVisS").textContent = "";
    $("evWinK").textContent = "Win probability · search before this move";
  }
  $("evDistV").textContent = dist.map((d, s) => d === null ? "×" : d).join(" · ");
  $("evDistS").textContent = "S · W · N · E";
}
function renderInfo() {
  const h = cur.h, rows = [];
  const add = (k, v) => rows.push(`<dt>${k}</dt><dd>${v}</dd>`);
  add("Result", resultFull(h));
  add("Plies", `${h.plies}${h.elims ? ` + ${h.elims} elimination${h.elims > 1 ? "s" : ""}` : ""}`);
  add("Rules", `${ruleLabel(h)}, maxPlies ${h.maxPlies}` + (wallsStd(h) ? "" : `, walls ${h.walls.join("/")}`));
  add("Type", escapeHtml(h.gtype) + (h.startTurnIdx ? ` · ${h.startTurnIdx} opening plies` : ""));
  if (h.table) add("Match", `${escapeHtml(h.table)} · opening ${h.opening} · rotation ${h.rotation}`);
  for (let s = 0; s < 4; s++) add(`<span class="who ${SEATS[s]}"></span> ${SEATS[s]}`, `${escapeHtml(h.names[s])} · ${escapeHtml(h.types[s] || "?")}${h.visits[s] ? " @" + h.visits[s] : ""}${modelOf(h, s) && modelOf(h, s) !== h.names[s] ? " · " + escapeHtml(modelOf(h, s)) : ""}`);
  if (h.gameHash) add("Hash", `<code>${h.gameHash}</code>`);
  $("info").innerHTML = rows.join("");
  $("chartLegend").innerHTML = ["S", "W", "N", "E"].map((n, i) => `<span class="sq" style="--c:${SEAT_COL[i]}">${n}</span>`).join("") + `<span class="sq" style="--c:${DRAW_COL}">draw</span>`;
}
function renderMoveList() {
  const rows = [];
  let marked = false;
  cur.events.forEach((e, i) => {
    if (!marked && cur.h.startTurnIdx > 0 && i === 0) { rows.push(`<div class="sep">opening (not played by the engines)</div>`); }
    if (cur.h.startTurnIdx > 0 && i === cur.h.startTurnIdx && !marked) { marked = true; rows.push(`<div class="sep">engine moves</div>`); }
    const init = i < cur.h.startTurnIdx;
    const label = e.elim ? "✕ eliminated" : e.text;
    const ev = e.comment ? `v${e.comment.visits}` + (e.comment.weight > 0 ? "" : " ·") : "";
    rows.push(`<div class="m${init ? " init" : ""}${i === curBlunder ? " blunder" : ""}" data-i="${i + 1}"><span class="n">${e.elim ? "" : e.ply}</span><span class="who ${SEATS[e.seat]}"></span><span class="mv">${label}</span><span class="ev">${ev}</span></div>`);
  });
  $("moveList").innerHTML = rows.join("");
}
function update(scroll = true) {
  if (!cur) return;
  ply = Math.max(0, Math.min(cur.events.length, ply));
  const dist = renderBoard().dist;
  renderCards(dist);
  renderEval(dist);
  $("moveSlider").value = ply;
  const last = cur.states[ply].last;
  $("moveInfo").textContent = ply === 0 ? `start · ${cur.events.length} events` : `${ply}/${cur.events.length} · ${last.elim ? "✕ " + SEATS[last.seat] + " eliminated" : SEATS[last.seat] + " " + cur.events[ply - 1].text}`;
  for (const r of document.querySelectorAll("#moveList .m")) r.classList.toggle("cur", +r.dataset.i === ply);
  const c = document.querySelector("#moveList .m.cur");
  if (c && scroll) scrollWithin($("moveList"), c);
  renderChart();
}

// ---- chart -------------------------------------------------------------------------------
function chartGeom() {
  const svg = $("chart"), W = svg.clientWidth || 320, H = 230, L = 34, R = 10, T = 10, B = 22;
  const n = Math.max(cur.events.length, 1);
  return { W, H, L, R, T, B, X: i => L + (W - L - R) * i / n, Y: p => T + (H - T - B) * (1 - p), n };
}
function chartPlyAt(e) {
  const r = $("chart").getBoundingClientRect(), g = chartGeom();
  return Math.max(0, Math.min(cur.events.length, Math.round((e.clientX - r.left - g.L) / (g.W - g.L - g.R) * g.n)));
}
function renderChart() {
  if (!cur) return;
  const svg = $("chart"), g = chartGeom();
  svg.setAttribute("viewBox", `0 0 ${g.W} ${g.H}`);
  svg.innerHTML = "";
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    el("line", { x1: g.L, x2: g.W - g.R, y1: g.Y(t), y2: g.Y(t), stroke: "#e8e4da", "stroke-width": 1 }, svg);
    txt(svg, g.L - 5, g.Y(t) + 3.5, Math.round(t * 100) + "%", { "text-anchor": "end" });
  }
  const step = Math.max(1, Math.ceil(cur.events.length / 8 / 10) * 10);
  for (let i = 0; i <= cur.events.length; i += step) txt(svg, g.X(i), g.H - 6, i, { "text-anchor": "middle" });
  if (cur.h.startTurnIdx > 0) el("rect", { x: g.X(0), y: g.T, width: g.X(Math.min(cur.h.startTurnIdx, cur.events.length)) - g.X(0), height: g.H - g.T - g.B, fill: "#00000008" }, svg);
  const series = [4, 0, 1, 2, 3];   // draw first so the seats are on top
  const win = cur.h.winner;
  for (const t of series) {
    const col = t === 4 ? DRAW_COL : SEAT_COL[t], emph = t === win;
    let d = "", pen = false;
    cur.events.forEach((e, i) => {
      if (!e.comment) { pen = false; return; }
      d += (pen ? "L" : "M") + g.X(i).toFixed(1) + " " + g.Y(e.comment.p[t]).toFixed(1);
      pen = true;
    });
    if (d) el("path", { d, fill: "none", stroke: col, "stroke-width": emph ? 3 : 1.6, opacity: emph ? 1 : (win >= 0 ? 0.7 : 0.9), "stroke-dasharray": t === 4 ? "4 3" : "none", "stroke-linejoin": "round" }, svg);
  }
  const hv = el("line", { x1: 0, x2: 0, y1: g.T, y2: g.H - g.B, stroke: "#9aa5b1", "stroke-width": 1, visibility: "hidden", "stroke-dasharray": "3 3" }, svg);
  svg._hover = hv;
  el("line", { x1: g.X(ply), x2: g.X(ply), y1: g.T, y2: g.H - g.B, stroke: "#1f2933", "stroke-width": 1.5 }, svg);
}
