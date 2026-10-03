"use strict";
// ---- Play GUI: talks to serve.py, renders its state, turns clicks into QTP moves -------------------------
const $ = id => document.getElementById(id);
const PRESETS = { easy: 1, normal: 16, hard: 256, max: 800 };
const LS_KEY = "kq-play-settings-v1";
const NAME = { b: "Black", w: "White" };
const GOAL_ROW = { b: 1, w: 9 };
const opp = c => (c === "b" ? "w" : "b");

let S = null;             // latest /api/state
let lastJson = "";
let viewPly = null;       // null: the current position; otherwise a position under review (read-only)
let bottom = "w";         // colour drawn at the bottom of the board
let orientedFor = null;   // game_id the orientation was chosen for
let selected = false;     // the human clicked their own pawn
let pending = false;      // a POST is in flight
let dismissedOver = null; // game-over overlay closed for this key
let autoStarted = false;
let connLost = false;
let pollTimer = null;
let movesKey = "";
let pvHover = null;       // {color, moves}: a candidate's line shown on the board instead of the candidates
let candKey = "";

const board = new Board($("board"));

// ---- settings (localStorage is only a convenience; everything works without it) -----------------------
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch (e) { return {}; }
}
function saveSettings(o) {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ ...loadSettings(), ...o })); } catch (e) { /* ignore */ }
}
let showEval = loadSettings().showEval !== false;
let polOnly = !!loadSettings().polOnly;
let pvOnHover = loadSettings().pvOnHover !== false;
// candidate list order: "lcb", "visits" or "order" (the engine's own ranking)
let sortBy = ["visits", "lcb", "order"].includes(loadSettings().sortBy) ? loadSettings().sortBy : "lcb";
const SORTS = {
  visits: (a, b) => b.visits - a.visits || a.order - b.order,
  lcb: (a, b) => (b.lcb ?? -9) - (a.lcb ?? -9) || a.order - b.order,
  order: (a, b) => a.order - b.order,
};

// ---- server -----------------------------------------------------------------------------------------------
async function api(path, body) {
  const opts = body === undefined ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  let data = null;
  try { data = await r.json(); } catch (e) { /* not JSON */ }
  if (!r.ok) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}

function schedulePoll(ms) {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(poll, ms);
}
async function poll() {
  try {
    setState(await api("api/state"));
    if (connLost) { connLost = false; render(); }
  } catch (e) {
    if (!connLost) { connLost = true; render(); }
  }
  const fast = S && (S.thinking || S.busy || S.engine.status === "starting" || S.analysis.running);
  schedulePoll(fast ? 250 : 1000);
}

function setState(s) {
  const j = JSON.stringify(s);
  if (j === lastJson) return;
  lastJson = j;
  const prev = S;
  S = s;
  if (orientedFor === null && S.started) orient();
  if (prev && prev.game_id !== S.game_id && S.moves.length <= 1) { viewPly = null; selected = false; }
  if (viewPly !== null && viewPly >= S.moves.length) viewPly = null;
  if (!autoStarted && S.engine.status === "ready" && !S.started) {
    autoStarted = true;
    const st = loadSettings();
    startGame(st.side || "b", visitsOf(st));
  }
  render();
}

function orient() {
  if (S.mode === "analysis" && orientedFor !== null) { orientedFor = S.game_id; return; }
  bottom = S.settings.human || "w";
  orientedFor = S.game_id;
}

async function act(fn, okMsg) {
  pending = true;
  render();
  try {
    const s = await fn();
    if (s && s.moves) setState(s);
    if (okMsg) toast(okMsg);
  } catch (e) {
    toast(e.message, true);
  } finally {
    pending = false;
    render();
    schedulePoll(150);
  }
}

function visitsOf(st) {
  if (st.preset === "custom") return Math.max(1, Math.min(100000, parseInt(st.custom, 10) || 64));
  return PRESETS[st.preset] || PRESETS.normal;
}

function startGame(side, visits) {
  viewPly = null; selected = false; dismissedOver = null; pvHover = null;
  return act(async () => {
    let s = await api("api/new", { human: side, visits });
    if (side === "analysis") s = await api("api/analyze", { on: true, max_visits: maxVisitsSetting() });
    S = null; lastJson = "";
    setState(s);
    orient();
    return s;
  });
}

function playMove(move) {
  selected = false; pvHover = null;
  board.setHover(null);
  return act(() => api("api/move", { move }));
}

// analysis mode
function maxVisitsSetting() {
  const v = parseInt(loadSettings().maxVisits, 10);
  return v > 0 ? v : 0;
}
function enterAnalysis() {
  viewPly = null; selected = false; pvHover = null;
  return act(async () => {
    await api("api/analysis_mode", {});
    return api("api/analyze", { on: true, max_visits: maxVisitsSetting() });
  });
}
function toggleAnalysis() {
  if (!isAnalysis()) return;
  act(() => api("api/analyze", { on: !S.analysis.on, max_visits: maxVisitsSetting() }));
}
function gotoPly(p) {
  if (!S || !isAnalysis()) return;
  const total = S.moves.length + S.redo.length;
  p = Math.max(0, Math.min(total, p));
  if (p === S.moves.length || S.busy || pending) return;
  selected = false; pvHover = null;
  board.setHover(null);
  act(() => api("api/goto", { ply: p }));
}

// ---- derived state ------------------------------------------------------------------------------------------
function curPly() { return viewPly === null ? S.moves.length : viewPly; }
function isAnalysis() { return !!S && S.mode === "analysis"; }
// the colour the board takes moves for: the human, or the side to move in analysis mode
function mover() { return isAnalysis() ? S.to_move : S.settings.human; }
function myTurn() {
  if (!S || !S.started || viewPly !== null || S.winner || S.busy || pending || S.engine.status !== "ready") return false;
  if (isAnalysis()) return true;
  return !!S.settings.human && S.to_move === S.settings.human && !S.thinking;
}
// the candidate list for the current position: the live analysis, else a hint
function currentCands() {
  if (!S || !S.started || viewPly !== null || S.winner) return null;
  const a = S.analysis.current;
  if (isAnalysis() && a && a.ply === S.moves.length && a.moves.length)
    return { src: "analysis", color: a.color, moves: a.moves, root: a.root };
  const h = S.hint;
  if (h && h.ply === S.moves.length) return { src: "hint", color: h.color, moves: h.moves, root: h.root };
  return null;
}
// the candidates drawn on the board: searched moves with a fair share of the visits, best first
function boardCands(c) {
  if (!c) return null;
  const searched = c.moves.filter(m => m.visits > 0).sort((a, b) => a.order - b.order);
  const top = searched.length ? Math.max(...searched.map(m => m.visits)) : 0;
  const moves = searched.filter((m, i) => i === 0 || m.visits >= Math.max(2, top * 0.002)).slice(0, 16);
  return { color: c.color, moves };
}
function playerLabel(c) {
  if (isAnalysis()) return NAME[c];
  const h = S.settings.human;
  if (!h) return "KataQuoridor";
  return c === h ? "You" : "KataQuoridor";
}
function leadText(e) {
  const l = e.black_lead, a = Math.abs(l);
  if (e.source === "final") return `${NAME[l > 0 ? "b" : "w"]} won by ${a.toFixed(0)} move${a === 1 ? "" : "s"}`;
  if (a < 0.05) return "Even race";
  return `${NAME[l > 0 ? "b" : "w"]} leads by ${a.toFixed(1)} move${a.toFixed(1) === "1.0" ? "" : "s"}`;
}
const pct = x => (100 * x).toFixed(1) + "%";

// ---- rendering ---------------------------------------------------------------------------------------------------
function render() {
  document.body.classList.toggle("noeval", !showEval);
  $("evalBtn").classList.toggle("on", showEval);
  $("pvBtn").classList.toggle("on", pvOnHover);
  renderBanner();
  renderEngineInfo();
  if (!S || !S.started) {
    board.draw(null);
    $("cardTop").innerHTML = ""; $("cardBottom").innerHTML = "";
    renderStatus();
    renderButtons();
    return;
  }
  const ply = curPly(), pos = S.positions[ply], interactive = myTurn();
  const lastMove = ply > 0 ? S.moves[ply - 1] : null;
  const legal = interactive ? new Set(S.legal_moves) : null;
  if (board.hover) board.hover.legal = !!(legal && legal.has(board.hover.move));
  const cands = currentCands();
  if (!cands) pvHover = null;
  board.bottom = bottom;
  board.draw({
    pos, lastMove, legal, interactive,
    prevPawn: lastMove ? S.positions[ply - 1].pawns[lastMove.color] : null,
    mover: mover(),
    cands: boardCands(cands),
    pv: pvHover,
    selected: selected && interactive,
  });
  renderCard($("cardTop"), opp(bottom), pos);
  renderCard($("cardBottom"), bottom, pos);
  renderStatus();
  renderEval(ply);
  renderCands(cands);
  renderMoves(ply);
  renderOverlay();
  renderButtons();
}

function renderEngineInfo() {
  let t = "Starting…";
  if (S) {
    const e = S.engine;
    if (e.status === "starting") t = "Loading the engine…";
    else if (e.status === "error") t = "Engine stopped";
    else t = [e.name && e.name.replace(/\s*\(based on.*\)/, ""), e.model,
      S.started && (isAnalysis() ? "analysis mode" : S.settings.visits + (S.settings.visits === 1 ? " visit" : " visits"))].filter(Boolean).join(" · ");
  }
  $("engineInfo").textContent = t;
}

function renderBanner() {
  const b = $("banner");
  if (connLost) {
    b.hidden = false;
    $("bannerTitle").textContent = "Lost the connection to the server.";
    $("bannerMsg").textContent = "Is serve.py still running? The page reconnects on its own.";
    $("restartBtn").hidden = true;
  } else if (S && S.engine.status === "error") {
    b.hidden = false;
    $("bannerTitle").textContent = "The engine stopped.";
    $("bannerMsg").textContent = S.engine.error || "";
    $("restartBtn").hidden = false;
  } else {
    b.hidden = true;
  }
}

function pips(left) {
  let s = "";
  for (let i = 0; i < 10; i++) s += `<i class="${i < left ? "" : "used"}"></i>`;
  return s;
}

function renderCard(box, c, pos) {
  const toMove = viewPly === null ? S.to_move : (curPly() % 2 === 0 ? "b" : "w");
  const over = viewPly === null ? !!S.winner : false;
  const isAi = !isAnalysis() && S.settings.human !== c;
  const thinking = isAi && S.thinking && viewPly === null && S.to_move === c;
  const d = pos.dist ? pos.dist[c] : null;
  const left = pos.walls_left ? pos.walls_left[c] : 10;
  box.style.setProperty("--acc", `var(--${c}-acc)`);
  box.classList.toggle("tomove", !over && toMove === c);
  box.innerHTML =
    `<svg width="28" height="28" aria-hidden="true"><circle cx="14" cy="14" r="10.5" fill="var(--${c}-pawn)" stroke="var(--${c}-acc)" stroke-width="3.5"/></svg>` +
    `<div class="who"><div class="name">${playerLabel(c)} <span class="sub">${NAME[c]}${c === "b" ? " · 1st" : ""}</span>` +
    (thinking ? ` <span class="thinking"><span class="spin"></span>thinking…</span>` : "") + `</div>` +
    `<div class="sub">→ row ${GOAL_ROW[c]} · <b>${d === null ? "?" : d}</b> step${d === 1 ? "" : "s"} to go · ${left} wall${left === 1 ? "" : "s"}</div></div>` +
    `<div class="walls" title="${left} walls left">${pips(left)}</div>`;
}

function renderStatus() {
  const st = $("status");
  let html = "";
  if (!S || S.engine.status === "starting") html = `<span class="spin"></span>Loading the engine and the net… (the first start can take a while)`;
  else if (S.engine.status === "error") html = "The engine stopped: use <b>Restart engine</b> above.";
  else if (!S.started) html = "Choose <b>New game</b> to start.";
  else if (viewPly !== null) html = `Reviewing the position after move ${viewPly} of ${S.moves.length} (read-only). <a href="#" id="stBack">Back to current</a>`;
  else if (isAnalysis()) html = analysisStatus();
  else if (S.winner) html = `<b>Game over:</b> ${NAME[S.winner]} wins by ${S.margin} move${S.margin === 1 ? "" : "s"}.`;
  else if (S.busy === "hint") html = `<span class="spin"></span>Searching for a hint (${S.settings.hint_visits} visits)…`;
  else if (S.thinking) html = `<span class="spin"></span>KataQuoridor is thinking… (${S.settings.visits} visit${S.settings.visits === 1 ? "" : "s"})` + (S.settings.human ? "" : " · AI vs AI demo");
  else if (S.ai_error) html = `<span class="err">${escapeHtml(S.ai_error)}</span>`;
  else if (myTurn()) {
    const left = S.walls_left[S.settings.human];
    html = `<b>Your move</b> (${NAME[S.settings.human]}): click a dot to move your pawn` +
      (left > 0 ? ", or a groove between cells to place a wall." : ". No walls left.");
  } else if (pending) html = `<span class="spin"></span>Sending…`;
  st.innerHTML = html;
  const back = $("stBack");
  if (back) back.onclick = e => { e.preventDefault(); review(null); };
  $("moveInput").disabled = !myTurn();
}

function analysisStatus() {
  if (S.winner) return `<b>Game over:</b> ${NAME[S.winner]} wins by ${S.margin} move${S.margin === 1 ? "" : "s"}. ← to step back.`;
  if (S.busy === "hint") return `<span class="spin"></span>Searching (${S.settings.hint_visits} visits)…`;
  if (pending || S.busy) return `<span class="spin"></span>Working…`;
  const a = S.analysis, cur = a.current, v = cur && cur.root ? cur.root.visits : 0;
  const cap = a.max_visits ? fmtVisits(a.max_visits) : "∞";
  let eng;
  if (!S.engine.can_analyze) eng = "this engine has no kata-analyze; use Hint";
  else if (a.running) eng = `<span class="spin"></span>analyzing · ${fmtVisits(v)} / ${cap} visits · Space pauses`;
  else if (a.on && cur && cur.done) eng = `analysis done · ${fmtVisits(v)} / ${cap} visits`;
  else eng = `analysis paused${v ? " · " + fmtVisits(v) + " visits" : ""} · Space starts`;
  return `<b>${NAME[S.to_move]} to move</b> · ${eng}`;
}

function renderEval(ply) {
  const e = S.positions[ply].eval;
  $("evalPly").textContent = ply === 0 ? "start position" : `after move ${ply}`;
  const h = S.settings.human;
  const you = c => (h === c ? " (you)" : "");
  if (!e) {
    $("evBarB").style.width = "50%";
    $("evB").textContent = "Black —"; $("evW").textContent = "White —";
    $("evLead").textContent = S.thinking ? "Evaluating…" : "No evaluation";
    $("evSrc").textContent = "";
  } else {
    $("evBarB").style.width = (100 * e.black_win).toFixed(1) + "%";
    $("evB").textContent = `Black${you("b")} ${pct(e.black_win)}`;
    $("evW").textContent = `White${you("w")} ${pct(e.white_win)}`;
    $("evLead").textContent = leadText(e);
    let src = e.source === "search" ? `Search · ${e.visits} visits` : e.source === "net" ? "Net estimate (no search)" : "Final result";
    if (e.pv && e.pv.length) src += ` · best line ${e.pv.slice(0, 6).join(" ")}${e.pv.length > 6 ? " …" : ""}`;
    $("evSrc").textContent = src;
  }
  const net = S.positions[ply].net;
  $("evNet").textContent = net && e && e.source !== "net"
    ? `Net alone: Black ${pct(net.black_win)} · ${leadText(net)}` : "";
  renderChart(ply);
}

function renderChart(ply) {
  const svg = $("chart");
  const W = Math.max(200, svg.clientWidth || 300), H = 84, L = 22, R = 6, T = 6, B = 6;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = "";
  const n = S.moves.length, span = Math.max(n, 10);
  const X = i => L + (i / span) * (W - L - R), Y = v => T + (1 - v) * (H - T - B);
  const mk = (tag, a) => el(tag, a, svg);
  mk("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "var(--chart-bg)", rx: 4 });
  mk("line", { x1: L, x2: W - R, y1: Y(0.5), y2: Y(0.5), stroke: "var(--line)", "stroke-dasharray": "3 3" });
  const lab = (y, s) => { const t = mk("text", { x: L - 4, y, "text-anchor": "end", "font-size": 9.5, fill: "var(--muted)" }); t.textContent = s; };
  lab(T + 8, "B"); lab(H - B - 1, "W");
  const pts = [];
  S.positions.forEach((p, i) => { if (p.eval) pts.push([X(i), Y(p.eval.black_win)]); });
  if (pts.length) {
    const d = pts.map((p, k) => (k ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("");
    mk("path", { d: d + `L${pts[pts.length - 1][0]} ${Y(0.5)}L${pts[0][0]} ${Y(0.5)}Z`, fill: "var(--b-pawn)", opacity: 0.08 });
    mk("path", { d, fill: "none", stroke: "var(--ink)", "stroke-width": 1.8, "stroke-linejoin": "round" });
  }
  mk("line", { x1: X(ply), x2: X(ply), y1: T, y2: H - B, stroke: "var(--b-acc)", "stroke-width": 2 });
  const e = S.positions[ply].eval;
  if (e) mk("circle", { cx: X(ply), cy: Y(e.black_win), r: 3.5, fill: "var(--b-acc)", stroke: "#fff", "stroke-width": 1.5 });
  svg.onclick = ev => {
    const r = svg.getBoundingClientRect();
    const x = (ev.clientX - r.left) * (W / r.width);
    review(Math.max(0, Math.min(n, Math.round(((x - L) / (W - L - R)) * span))));
  };
}

function signed(x, d = 1) { return (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d); }
const pct1 = x => x == null ? "—" : (100 * x).toFixed(1);

function renderCands(c) {
  const card = $("candCard");
  $("polOnly").checked = polOnly;
  if (!c) { card.hidden = true; candKey = ""; return; }
  card.hidden = false;
  const col = c.color, r = c.root;
  $("candTitle").textContent = c.src === "analysis" ? "Analysis" : "Hint";
  $("candSub").textContent = `${NAME[col]} to move` + (r ? ` · ${fmtVisits(r.visits)} visits` : "");
  // root summary
  let root = "";
  if (r) {
    root += `<div><b>${NAME[col]} ${pct1(r.winrate)}%</b> · lead ${signed(r.lead)} · utility ${signed(r.utility, 3)} · σ ${r.score_stdev == null ? "—" : r.score_stdev.toFixed(2)}</div>`;
    if (r.raw_wr_error != null)
      root += `<div class="muted">net uncertainty: winrate ±${r.raw_wr_error.toFixed(3)} · score ±${(r.raw_score_error || 0).toFixed(2)}` +
        (r.raw_time_left != null ? ` · time left ${r.raw_time_left.toFixed(1)}` : "") + `</div>`;
  }
  const net = S.positions[S.moves.length].net;
  if (net) {
    const nw = col === "b" ? net.black_win : net.white_win, nl = col === "b" ? net.black_lead : -net.black_lead;
    root += `<div class="muted">net value: ${NAME[col]} ${pct1(nw)}% · lead ${signed(nl)}</div>`;
  }
  $("candRoot").innerHTML = root;

  const searched = c.moves.filter(m => m.visits > 0).sort(SORTS[sortBy]);
  const total = searched.reduce((t, m) => t + m.visits, 0) || 1;
  let rows = searched;
  if (polOnly) rows = rows.concat(c.moves.filter(m => !(m.visits > 0)).sort((a, b) => (b.prior || 0) - (a.prior || 0)));
  const key = c.src + "|" + polOnly + "|" + sortBy + "|" + rows.map(m => m.move + m.visits).join(",") ;
  if (key === candKey) return;
  candKey = key;
  const sh = (k, label, title) => `<span class="sort${sortBy === k ? " on" : ""}" data-sort="${k}" title="${title} · click to sort">${label}${sortBy === k ? " ▾" : ""}</span>`;
  const head = `<div class="crow chead">${sh("order", "#", "The engine's ranking")}<span>move</span><span title="Win chance for ${NAME[col]} (side to move)">win%</span>` +
    `<span title="Expected margin in moves for ${NAME[col]}">lead</span>${sh("visits", "visits", "Visits (share of all child visits)")}` +
    `<span title="Raw policy prior of the net">policy</span>${sh("lcb", "lcb", "Lower confidence bound of the winrate")}</div>`;
  $("candList").innerHTML = head + rows.map(m => {
    const v = m.visits > 0;
    const cls = !v ? "unv" : m.order === 0 ? "top" : "";
    const pvs = v && m.pv ? m.pv.slice(1, 12).join(" ") : "";
    return `<div class="crow ${cls}${m.chosen ? " chosen" : ""}" data-move="${m.move}">` +
      `<span class="ci"><i></i>${v ? m.order + 1 : ""}</span><span class="mv">${m.move}</span>` +
      `<span>${v ? pct1(m.winrate) : "—"}</span><span>${v ? signed(m.lead) : "—"}</span>` +
      `<span>${v ? fmtVisits(m.visits) + ` <small>${(100 * m.visits / total).toFixed(0)}%</small>` : "0"}</span>` +
      `<span>${m.prior == null ? "—" : (100 * m.prior).toFixed(m.prior < 0.001 ? 3 : 1) + "%"}</span>` +
      `<span>${v && m.lcb != null ? pct1(Math.max(0, m.lcb)) : "—"}</span>` +
      (pvs ? `<span class="cpv">${pvs}</span>` : "") + `</div>`;
  }).join("") + `<div class="note">Values for ${NAME[col]}, the side to move. Hover a line to see its continuation; click to play it.</div>`;
  card.querySelectorAll(".crow:not(.chead)").forEach(row => {
    const m = c.moves.find(x => x.move === row.dataset.move);
    const alpha = m && m.visits > 0 ? (m.order === 0 ? 1 : candAlpha(m.visits, searched[0] ? Math.max(...searched.map(x => x.visits)) : 0)) : 0;
    row.style.setProperty("--dot", m && m.order === 0 && m.visits > 0 ? CAND_TOP : CAND_OFF);
    row.style.setProperty("--dot-a", alpha);
  });
}

function renderMoves(ply) {
  const key = [S.game_id, S.moves.length, showEval, S.redo.map(m => m.move).join(","), S.positions.map(p => p.eval ? p.eval.black_win.toFixed(3) : "-").join(",")].join("|");
  if (key !== movesKey) {
    movesKey = key;
    const e0 = S.positions[0].eval;
    const rows = [`<div class="m" data-ply="0"><span class="n"></span><span class="who"></span><span class="mv muted">start</span><span class="ev">${e0 ? "B " + (100 * e0.black_win).toFixed(0) + "%" : ""}</span></div>`];
    S.moves.forEach((m, i) => {
      const e = S.positions[i + 1].eval;
      const wall = m.move.length === 3;
      rows.push(`<div class="m" data-ply="${i + 1}"><span class="n">${i + 1}</span><span class="who ${m.color}"></span>` +
        `<span class="mv${wall ? " wall" : ""}">${m.move}</span>` +
        `<span class="ev">${e ? "B " + (100 * e.black_win).toFixed(0) + "%" : ""}</span></div>`);
    });
    S.redo.forEach((m, i) => {
      const n = S.moves.length + i + 1;
      rows.push(`<div class="m redo" data-ply="${n}" title="Undone: click to go back to it"><span class="n">${n}</span><span class="who ${m.color}"></span>` +
        `<span class="mv${m.move.length === 3 ? " wall" : ""}">${m.move}</span><span class="ev"></span></div>`);
    });
    $("moveList").innerHTML = rows.join("");
  }
  $("moveCount").textContent = S.moves.length ? `${S.moves.length} played` + (S.redo.length ? ` · ${S.redo.length} undone` : "") : "";
  document.querySelectorAll("#moveList .m.cur").forEach(x => x.classList.remove("cur"));
  const row = document.querySelector(`#moveList .m[data-ply="${ply}"]`);
  if (row) {
    row.classList.add("cur");
    // scroll only the list (scrollIntoView would also scroll the page)
    const list = $("moveList");
    if (row.offsetTop < list.scrollTop) list.scrollTop = row.offsetTop;
    else if (row.offsetTop + row.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = row.offsetTop + row.offsetHeight - list.clientHeight;
  }
  $("reviewBar").hidden = viewPly === null;
  $("reviewText").textContent = viewPly === null ? "" : `Move ${viewPly} of ${S.moves.length}`;
}

function renderOverlay() {
  const key = S.game_id + ":" + S.moves.length;
  const show = !!S.winner && viewPly === null && dismissedOver !== key && !isAnalysis();
  $("overlay").hidden = !show;
  if (!show) return;
  $("ovTitle").textContent = `${NAME[S.winner]} wins by ${S.margin} move${S.margin === 1 ? "" : "s"}`;
  const h = S.settings.human;
  $("ovSub").textContent = !h ? "AI vs AI demo finished." : S.winner === h ? "Well played, you win!" : "KataQuoridor wins this one.";
  $("overlay").className = "win-" + S.winner;
}

function renderButtons() {
  const ready = S && S.engine.status === "ready";
  const idle = ready && S.started && !S.thinking && !S.busy && !pending;
  const h = S && S.settings.human;
  const ana = isAnalysis();
  $("undoBtn").disabled = !(idle && (ana ? S.moves.length > 0 : h && S.moves.some(m => m.color === h)));
  $("undoBtn").title = ana ? "Take back the last move (U, ←)" : "Take back your last move and the AI's reply (U)";
  $("hintBtn").disabled = !(idle && !S.winner && (ana ? !S.analysis.on : S.to_move === h));
  $("sgfBtn").classList.toggle("disabled", !idle);
  $("newBtn").disabled = !ready || pending;
  $("anaModeBtn").hidden = ana;
  $("anaModeBtn").disabled = !(ready && S.started && !S.busy && !pending);
  $("anaCtl").hidden = !ana;
  if (ana) {
    const on = S.analysis.on;
    $("anaBtn").classList.toggle("on", on);
    $("anaBtn").textContent = on ? "❚❚ Analysis" : "▶ Analysis";
    $("anaBtn").disabled = !ready || !S.engine.can_analyze;
    const mv = $("maxVisits");
    if (document.activeElement !== mv) mv.value = S.analysis.max_visits ? S.analysis.max_visits : "";
  }
  $("keys").textContent = ana
    ? "Space analysis · ← → Home End move through the game · U undo · H hint · wheel on board ↑↓ · P PV on hover · F flip · E evaluation · N new"
    : "N new · U undo · H hint · A analyze · P PV on hover · F flip · E evaluation · ← → / wheel review · Esc back";
}

function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

let toastTimer = null;
function toast(msg, bad) {
  const t = $("toast");
  t.textContent = msg;
  t.className = bad ? "bad" : "";
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 4500 : 2500);
}

function review(p) {
  if (!S || !S.started) return;
  if (isAnalysis()) return gotoPly(p === null ? S.moves.length + S.redo.length : p);
  viewPly = p === null || p >= S.moves.length ? null : Math.max(0, p);
  selected = false;
  board.setHover(null);
  render();
}

// ---- board input ----------------------------------------------------------------------------------------------
function hitAt(e) {
  if (!myTurn()) return null;
  const p = board.eventPoint(e), h = mover();
  const hit = board.hitTest(p.x, p.y, S.walls_left[h] > 0);
  if (!hit) return null;
  hit.legal = S.legal_moves.includes(hit.move);
  hit.own = hit.kind === "cell" && S.pawns[h] === hit.move;
  return hit;
}
const svgEl = $("board");
svgEl.addEventListener("pointermove", e => {
  if (e.pointerType === "touch") return;
  const hit = hitAt(e);
  board.setHover(hit);
  svgEl.style.cursor = !hit ? "" : hit.legal || hit.own ? "pointer" : hit.kind === "wall" ? "not-allowed" : "";
  setPvHover(hit && hit.legal ? hit.move : null);
});
// mouse wheel over the board: up = previous move, down = next move
let wheelAcc = 0;
svgEl.addEventListener("wheel", e => {
  if (!S || !S.started) return;
  e.preventDefault();
  wheelAcc += e.deltaY;
  if (Math.abs(wheelAcc) < 40) return;   // trackpads send many small deltas
  const step = wheelAcc > 0 ? 1 : -1;
  wheelAcc = 0;
  if (isAnalysis()) { if (!pending && !S.busy) gotoPly(S.moves.length + step); }
  else review(curPly() + step);
}, { passive: false });
svgEl.addEventListener("pointerleave", () => { board.setHover(null); svgEl.style.cursor = ""; setPvHover(null); });

// show a candidate's principal variation on the board while it is hovered (board or list)
function setPvHover(move) {
  if (!pvOnHover) move = null;
  const c = currentCands();
  const m = move && c ? c.moves.find(x => x.move === move && x.visits > 0 && x.pv && x.pv.length) : null;
  const next = m ? { color: c.color, moves: m.pv } : null;
  if ((next && pvHover && next.moves.join() === pvHover.moves.join()) || (!next && !pvHover)) return;
  pvHover = next;
  render();
}
// pointerdown + pointerup instead of click: the board is redrawn several times a second while the analysis
// runs, and a click whose target was replaced between press and release is never delivered
let downMove = null;
svgEl.addEventListener("pointerdown", e => {
  const hit = e.button === 0 ? hitAt(e) : null;
  downMove = hit ? hit.move : null;
});
svgEl.addEventListener("pointerup", e => {
  const hit = hitAt(e);
  const same = hit && hit.move === downMove;
  downMove = null;
  if (!same) return;
  // On touch screens the first tap only previews; a second tap on the same target plays it.
  if (e.pointerType === "touch" && hit.kind === "wall") {
    const h = board.hover;
    if (!h || h.move !== hit.move) { board.setHover(hit); return; }
  }
  if (hit.legal) return playMove(hit.move);
  if (hit.own) { selected = !selected; render(); return; }
  if (selected) { selected = false; render(); }
  if (hit.kind === "wall") toast(`${hit.move} is not a legal wall here`, true);
});

// ---- controls -------------------------------------------------------------------------------------------------
$("moveForm").addEventListener("submit", e => {
  e.preventDefault();
  const mv = $("moveInput").value.trim().toLowerCase();
  if (!mv) return;
  if (!myTurn()) return toast("It is not your turn.", true);
  if (!S.legal_moves.includes(mv)) {
    toast(/^([a-i][1-9]|[a-h][1-8][hv])$/.test(mv) ? `${mv} is not legal here` : `"${mv}" is not QTP notation (e.g. e8, e2h, d5v)`, true);
    return;
  }
  $("moveInput").value = "";
  playMove(mv);
});

// pointerdown, not click: the list is rebuilt several times a second while the analysis runs, which can
// swallow a click between press and release
$("candList").addEventListener("pointerdown", e => {
  if (e.button !== 0) return;
  const sc = e.target.closest("[data-sort]");
  if (sc) { sortBy = sc.dataset.sort; saveSettings({ sortBy }); candKey = ""; render(); return; }
  const row = e.target.closest(".crow");
  if (row && row.dataset.move && myTurn() && S.legal_moves.includes(row.dataset.move)) playMove(row.dataset.move);
});
$("candList").addEventListener("mouseover", e => {
  const row = e.target.closest(".crow");
  setPvHover(row ? row.dataset.move : null);
});
$("candList").addEventListener("mouseleave", () => setPvHover(null));
$("polOnly").addEventListener("change", e => { polOnly = e.target.checked; saveSettings({ polOnly }); candKey = ""; render(); });
$("moveList").addEventListener("click", e => {
  const row = e.target.closest(".m");
  if (row) review(+row.dataset.ply);
});
$("anaModeBtn").onclick = () => enterAnalysis();
$("anaBtn").onclick = () => toggleAnalysis();
$("maxVisits").addEventListener("change", e => {
  const v = parseInt(e.target.value, 10);
  const max = v > 0 ? v : 0;
  saveSettings({ maxVisits: max });
  if (isAnalysis()) act(() => api("api/analyze", { max_visits: max }));
});
$("backBtn").onclick = () => review(null);
$("undoBtn").onclick = () => { viewPly = null; act(() => api("api/undo", {})); };
$("hintBtn").onclick = () => { viewPly = null; act(() => api("api/hint", {}).then(() => api("api/state"))); };
$("flipBtn").onclick = () => { bottom = opp(bottom); board.setHover(null); render(); };
$("pvBtn").onclick = () => { pvOnHover = !pvOnHover; saveSettings({ pvOnHover }); if (!pvOnHover) pvHover = null; render(); };
$("evalBtn").onclick = () => { showEval = !showEval; saveSettings({ showEval }); movesKey = ""; render(); };
$("restartBtn").onclick = () => act(() => api("api/restart", {}).then(() => api("api/state")));
$("ovNew").onclick = () => openNewDialog();
$("ovReview").onclick = () => { dismissedOver = S.game_id + ":" + S.moves.length; render(); };

$("sgfBtn").addEventListener("click", async e => {
  e.preventDefault();
  if ($("sgfBtn").classList.contains("disabled")) return;
  try {
    const r = await fetch("api/sgf", { cache: "no-store" });
    if (!r.ok) { let m = `HTTP ${r.status}`; try { m = (await r.json()).error; } catch (x) { /* */ } throw new Error(m); }
    const name = (/filename="([^"]+)"/.exec(r.headers.get("Content-Disposition") || "") || [])[1] || "kataquoridor.sgfs";
    const url = URL.createObjectURL(await r.blob());
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    toast(`Saved ${name}`);
  } catch (err) {
    toast("SGF: " + err.message, true);
  }
});

// new-game dialog
const dlg = $("newDlg");
function openNewDialog() {
  const st = loadSettings();
  const side = st.side || "b", preset = st.preset || "normal";
  dlg.querySelectorAll('input[name="side"]').forEach(i => { i.checked = i.value === side; });
  dlg.querySelectorAll('input[name="preset"]').forEach(i => { i.checked = i.value === preset; });
  if (st.custom) $("customVisits").value = st.custom;
  dlg.showModal();
  $("startBtn").focus();
}
$("newBtn").onclick = openNewDialog;
$("customVisits").addEventListener("focus", () => { dlg.querySelector('input[value="custom"]').checked = true; });
dlg.addEventListener("close", () => {
  if (dlg.returnValue !== "ok") return;
  const side = (dlg.querySelector('input[name="side"]:checked') || {}).value || "b";
  const preset = (dlg.querySelector('input[name="preset"]:checked') || {}).value || "normal";
  const custom = parseInt($("customVisits").value, 10) || 64;
  saveSettings({ side, preset, custom });
  startGame(side, visitsOf({ preset, custom }));
});

document.addEventListener("keydown", e => {
  if (dlg.open || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
    if (e.key === "Escape" || (e.key === "Enter" && e.target.id === "maxVisits")) e.target.blur();
    return;
  }
  const k = e.key;
  if (k === " ") { e.preventDefault(); if (e.target.blur) e.target.blur(); toggleAnalysis(); return; }
  if ((k === "a" || k === "A") && !isAnalysis()) { if (!$("anaModeBtn").disabled) enterAnalysis(); return; }
  if (k === "Backspace" && isAnalysis()) { e.preventDefault(); if (!$("undoBtn").disabled) $("undoBtn").click(); return; }
  if (k === "n" || k === "N") { e.preventDefault(); if (!$("newBtn").disabled) openNewDialog(); }
  else if (k === "u" || k === "U") { if (!$("undoBtn").disabled) $("undoBtn").click(); }
  else if (k === "h" || k === "H") { if (!$("hintBtn").disabled) $("hintBtn").click(); }
  else if (k === "f" || k === "F") $("flipBtn").click();
  else if (k === "e" || k === "E") $("evalBtn").click();
  else if (k === "p" || k === "P") $("pvBtn").click();
  else if (k === "ArrowLeft" && S && S.started) { e.preventDefault(); review(curPly() - 1); }
  else if (k === "ArrowRight" && S && S.started) { e.preventDefault(); review(curPly() + 1); }
  else if (k === "Home" && S && S.started) review(0);
  else if (k === "End" && S && S.started) review(null);
  else if (k === "Escape") { if (selected) { selected = false; render(); } else if (!isAnalysis()) review(null); }
});
window.addEventListener("resize", () => { if (S && S.started) renderChart(curPly()); });

render();
poll();
