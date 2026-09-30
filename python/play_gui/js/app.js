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

const board = new Board($("board"));

// ---- settings (localStorage is only a convenience; everything works without it) -----------------------
function loadSettings() {
  try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch (e) { return {}; }
}
function saveSettings(o) {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ ...loadSettings(), ...o })); } catch (e) { /* ignore */ }
}
let showEval = loadSettings().showEval !== false;

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
  const fast = S && (S.thinking || S.busy || S.engine.status === "starting");
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
  viewPly = null; selected = false; dismissedOver = null;
  return act(async () => {
    const s = await api("api/new", { human: side, visits });
    S = null; lastJson = "";
    setState(s);
    orient();
    return s;
  });
}

function playMove(move) {
  selected = false;
  board.setHover(null);
  return act(() => api("api/move", { move }));
}

// ---- derived state ------------------------------------------------------------------------------------------
function curPly() { return viewPly === null ? S.moves.length : viewPly; }
function myTurn() {
  return !!S && S.started && viewPly === null && !S.winner && S.settings.human && S.to_move === S.settings.human &&
    !S.thinking && !S.busy && !pending && S.engine.status === "ready";
}
function playerLabel(c) {
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
  board.bottom = bottom;
  board.draw({
    pos, lastMove, legal, interactive,
    prevPawn: lastMove ? S.positions[ply - 1].pawns[lastMove.color] : null,
    human: S.settings.human,
    hints: viewPly === null && S.hint && S.hint.ply === S.moves.length ? S.hint.moves : [],
    selected: selected && interactive,
  });
  renderCard($("cardTop"), opp(bottom), pos);
  renderCard($("cardBottom"), bottom, pos);
  renderStatus();
  renderEval(ply);
  renderHint();
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
    else t = [e.name && e.name.replace(/\s*\(based on.*\)/, ""), e.model, S.started && S.settings.visits + (S.settings.visits === 1 ? " visit" : " visits")].filter(Boolean).join(" · ");
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
  const isAi = S.settings.human !== c;
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

function renderHint() {
  const h = S.hint, card = $("hintCard");
  if (!h || h.ply !== S.moves.length || viewPly !== null || S.winner) { card.hidden = true; return; }
  card.hidden = false;
  const c = h.color;
  $("hintSub").textContent = `${h.visits} visits`;
  $("hintList").innerHTML = h.moves.map((m, i) => {
    const win = c === "b" ? m.black_win : m.white_win, lead = c === "b" ? m.black_lead : -m.black_lead;
    return `<div class="hrow" data-move="${m.move}" title="Play ${m.move}"><span class="hn">${i + 1}</span><span class="mv">${m.move}</span>` +
      `<span class="hv">${pct(win)} · ${lead >= 0 ? "+" : "−"}${Math.abs(lead).toFixed(1)}</span>` +
      `<span class="pv">${(m.pv || []).slice(1, 5).join(" ")}</span></div>`;
  }).join("") + `<div class="note">Win chance and lead for you. Click a line to play its move.</div>`;
}

function renderMoves(ply) {
  const key = [S.game_id, S.moves.length, showEval, S.positions.map(p => p.eval ? p.eval.black_win.toFixed(3) : "-").join(",")].join("|");
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
    $("moveList").innerHTML = rows.join("");
  }
  $("moveCount").textContent = S.moves.length ? `${S.moves.length} played` : "";
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
  const show = !!S.winner && viewPly === null && dismissedOver !== key;
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
  $("undoBtn").disabled = !(idle && h && S.moves.some(m => m.color === h));
  $("hintBtn").disabled = !(idle && S.to_move === h && !S.winner);
  $("sgfBtn").classList.toggle("disabled", !idle);
  $("newBtn").disabled = !ready || pending;
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
  viewPly = p === null || p >= S.moves.length ? null : Math.max(0, p);
  selected = false;
  board.setHover(null);
  render();
}

// ---- board input ----------------------------------------------------------------------------------------------
function hitAt(e) {
  if (!myTurn()) return null;
  const p = board.eventPoint(e), h = S.settings.human;
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
});
svgEl.addEventListener("pointerleave", () => { board.setHover(null); svgEl.style.cursor = ""; });
svgEl.addEventListener("click", e => {
  const hit = hitAt(e);
  if (!hit) return;
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

$("hintList").addEventListener("click", e => {
  const row = e.target.closest(".hrow");
  if (row && myTurn() && S.legal_moves.includes(row.dataset.move)) playMove(row.dataset.move);
});
$("moveList").addEventListener("click", e => {
  const row = e.target.closest(".m");
  if (row) review(+row.dataset.ply);
});
$("backBtn").onclick = () => review(null);
$("undoBtn").onclick = () => { viewPly = null; act(() => api("api/undo", {})); };
$("hintBtn").onclick = () => { viewPly = null; act(() => api("api/hint", {}).then(() => api("api/state"))); };
$("flipBtn").onclick = () => { bottom = opp(bottom); board.setHover(null); render(); };
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
    if (e.key === "Escape") e.target.blur();
    return;
  }
  const k = e.key;
  if (k === "n" || k === "N") { e.preventDefault(); if (!$("newBtn").disabled) openNewDialog(); }
  else if (k === "u" || k === "U") { if (!$("undoBtn").disabled) $("undoBtn").click(); }
  else if (k === "h" || k === "H") { if (!$("hintBtn").disabled) $("hintBtn").click(); }
  else if (k === "f" || k === "F") $("flipBtn").click();
  else if (k === "e" || k === "E") $("evalBtn").click();
  else if (k === "ArrowLeft" && S && S.started) { e.preventDefault(); review(curPly() - 1); }
  else if (k === "ArrowRight" && S && S.started) { e.preventDefault(); review(curPly() + 1); }
  else if (k === "Home" && S && S.started) review(0);
  else if (k === "End") review(null);
  else if (k === "Escape") { if (selected) { selected = false; render(); } else review(null); }
});
window.addEventListener("resize", () => { if (S && S.started) renderChart(curPly()); });

render();
poll();
