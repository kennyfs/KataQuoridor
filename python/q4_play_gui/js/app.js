/* The page of the Q4 play GUI: polls /api/state, draws it, sends the user's actions. The engine decides every rule;
 * this file only shows the answers. Layout and behaviour follow the Duel page (python/play_gui/js/app.js). Text
 * that comes from the engine or from a pasted record goes in with textContent or through esc(). */
(function () {
  const C = window.Q4Core;
  const Q4Board = window.Q4Board;
  const U = window.Q4BoardUtil;
  // One id per browser tab (kept in sessionStorage, so a reload of the same tab takes the game over again at once).
  const CLIENT = (function () {
    const fresh = "tab" + Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    try {
      const kept = sessionStorage.getItem("q4play-client");
      if (kept && /^[A-Za-z0-9_-]{4,64}$/.test(kept)) return kept;
      sessionStorage.setItem("q4play-client", fresh);
    } catch (e) { /* no storage: a fresh id for this load */ }
    return fresh;
  })();
  const $ = (id) => document.getElementById(id);
  const SEAT = C.SEAT_LETTERS;
  const BOTS = ["greedy", "basher", "randomPawn", "random", "grudge"];
  const PRESETS = { easy: 1, normal: 16, hard: 256, max: 800 };
  const LS_KEY = "kq-q4play-settings-v1";
  // The candidates shown on the board and in the table: at most SHOWN_CANDIDATES searched moves whose visits are more
  // than MIN_VISIT_SHARE of all searched children's visits, best LCB first (see shownCandidates).
  const SHOWN_CANDIDATES = 5;
  const MIN_VISIT_SHARE = 0.01;
  const SVGNS = "http://www.w3.org/2000/svg";

  let state = null;
  let lastJson = "";
  let rot = 0;
  let busy = false;
  let busyText = "";
  let viewPly = null;               // null: the engine's position; otherwise a position under read-only review (play mode)
  let selectedSeat = null;          // a pawn clicked on the board (any seat)
  let pendingLegal = null;          // {seat, legal} for a seat other than the one to move
  let pvOverlay = null;             // the hovered candidate's principal variation: [{seat, move}]
  let pvKey = "";
  let lockedOut = false;
  let connLost = false;
  let connMsg = "";
  let aiRunning = false;
  let dismissedOver = null;
  let board = null;
  let seatCards = [];
  let candKey = "";
  let movesKey = "";
  let lastShownCur = -1;

  // -- settings (localStorage is only a convenience; everything works without it) ----------------------
  function loadSettings() {
    try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch (e) { return {}; }
  }
  function saveSettings(o) {
    try { localStorage.setItem(LS_KEY, JSON.stringify(Object.assign(loadSettings(), o))); } catch (e) { /* ignore */ }
  }
  let showEval = loadSettings().showEval !== false;
  let pvOnHover = loadSettings().pvOnHover !== false;
  const SORTS = ["lcb", "visits", "win", "prior", "utility"];
  let sortKey = SORTS.includes(loadSettings().sortKey) ? loadSettings().sortKey : "lcb";

  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
  function svgEl(name, attrs, parent) {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  // -- transport ---------------------------------------------------------------------------------------
  async function api(method, path, body) {
    const opts = { method, headers: { "X-Client": CLIENT } };
    if (method === "POST") {   // every POST is JSON (the server refuses anything else), "{}" when there is nothing to send
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body === undefined ? {} : body);
    }
    let r;
    try {
      r = await fetch(path, opts);
    } catch (e) {
      throw failure("cannot reach the local server (is serve.py still running?)", 0);
    }
    const text = await r.text();
    let data = {};
    try { data = JSON.parse(text); } catch (e) { data = { error: text }; }
    if (!r.ok) {
      const err = failure(data.error || ("HTTP " + r.status), r.status);
      err.engineDown = !!data.engine_down;
      throw err;
    }
    return data;
  }

  function failure(message, status) {
    const e = new Error(message);
    e.status = status;
    return e;
  }

  function setState(s) {
    lockedOut = false;
    connLost = false;
    const j = JSON.stringify(s);
    if (j === lastJson) return false;
    lastJson = j;
    state = s;
    if (viewPly !== null && (viewPly > state.length || viewPly === state.cursor)) viewPly = null;
    render();
    return true;
  }

  async function refresh() {
    const s = await api("GET", "/api/state");
    const changed = setState(s);
    if (!changed) renderBanner();
    return s;
  }

  // Runs one action: busy flag, toast on error, state refreshed afterwards. An action started while another one
  // is running is dropped (the wheel and the arrow keys rely on this: one step per engine answer).
  async function act(text, fn) {
    if (busy) return undefined;
    busy = true;
    busyText = text;
    document.body.style.cursor = "progress";
    render();
    try {
      const r = await fn();
      await refresh();
      return r;
    } catch (e) {
      toast(e.message, true);
      if (e.status === 423) lockedOut = true;
      else if (e.status === 0) { connLost = true; connMsg = e.message; }
      try { await refresh(); } catch (e2) { /* the server may be gone: the banner says so */ }
      return undefined;
    } finally {
      busy = false;
      document.body.style.cursor = "";
      render();
    }
  }

  async function poll() {
    try {
      await refresh();
    } catch (e) {
      if (e.status === 423) lockedOut = true;
      else if (e.status === 400) { connLost = true; connMsg = "This page did not identify itself; reload it."; }
      else { connLost = true; connMsg = e.message; }
      renderBanner();
    }
    const fast = state && (busy || (state.analysis && state.analysis.running));
    setTimeout(poll, fast ? 300 : 1000);
  }

  // -- messages -------------------------------------------------------------------------------------------
  let toastTimer = null;
  function toast(text, bad) {
    const t = $("toast");
    t.textContent = text;
    t.className = bad ? "bad" : "";
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, bad ? 4500 : 2500);
  }

  function askConfirm(title, text, ok) {
    return new Promise((resolve) => {
      const d = $("confirmDlg");
      $("confirmTitle").textContent = title;
      $("confirmText").textContent = text;
      $("confirmGo").textContent = ok;
      d.returnValue = "";
      d.addEventListener("close", () => resolve(d.returnValue === "ok"), { once: true });
      d.showModal();
      $("confirmGo").focus();
    });
  }

  // -- derived state ------------------------------------------------------------------------------------
  function isAnalysis() { return !!state && state.mode === "analysis"; }
  function reviewing() { return viewPly !== null && !!state && viewPly !== state.cursor; }
  function curPly() { return viewPly === null ? state.cursor : viewPly; }
  function ready() { return !!state && state.engine.status === "ready"; }
  // The position shown: the engine's, or (read-only review) rebuilt from the event list.
  function shownPosition() {
    return reviewing() ? C.replay(state.events, viewPly, state.rules.initialWalls) : state.position;
  }
  function isAiToMove() {
    const pos = state && state.position;
    return !!pos && !pos.finished && state.seats[pos.toMove].kind !== "human";
  }
  // may the user move on the board: the engine's position, the game still running
  function canPlay() { return ready() && !busy && !reviewing() && !state.position.finished; }

  function kindLabel(seat) {
    if (seat.kind === "human") return "human";
    if (seat.kind === "net") return "AI net, " + (seat.visits || state.settings.maxVisits) + " visits";
    return "bot " + seat.bot + (seat.target !== null && seat.target !== undefined ? " → " + SEAT[seat.target] : "");
  }

  // -- the view of the board ----------------------------------------------------------------------------
  function shownCandidates(moves, mover) {
    const total = moves.reduce((t, m) => t + (m.visits > 0 ? m.visits : 0), 0) || 1;
    return moves.filter(m => m.visits > 0 && m.visits / total > MIN_VISIT_SHARE && C.parseMove(m.move))
      .map(m => ({
        move: m.move, visits: m.visits, lcb: m.lcb, prior: m.prior, utility: m.utility,
        win: m.value ? m.value[mover] : null, value: m.value, pv: m.pv,
      }))
      .sort((a, b) => (b.lcb - a.lcb) || (b.visits - a.visits))
      .slice(0, SHOWN_CANDIDATES);
  }

  function currentCandidates() {
    if (!state || reviewing()) return [];
    const an = state.analysis, pos = state.position;
    if (an.on && an.moves && an.moves.length && an.ply === state.cursor && !pos.finished) return shownCandidates(an.moves, pos.toMove);
    return [];
  }

  function buildView() {
    const rv = reviewing(), n = curPly();
    const pos = shownPosition();
    const events = state.events.slice(0, n);
    const last = events.length ? events[events.length - 1] : null;
    const lastMove = last && !last.elim && last.a ? C.parseMove(last.a) : null;
    const dotSeat = selectedSeat !== null && !rv ? selectedSeat : pos.toMove;
    let legal = [];
    if (!rv) {
      if (dotSeat === pos.toMove) legal = pos.legal || [];
      else if (pendingLegal && pendingLegal.seat === dotSeat) legal = pendingLegal.legal;
    }
    const legalPawns = new Set(legal.filter(m => C.parseMove(m) && C.parseMove(m).kind === "pawn"));
    // Walls can only be placed by the seat to move: the wall preview is for that seat.
    const canWall = !rv && !pos.finished && dotSeat === pos.toMove && pos.wallsLeft[pos.toMove] > 0;
    const legalWalls = new Set(canWall ? (pos.legal || []).filter(m => m.endsWith("h") || m.endsWith("v")) : []);
    return {
      position: pos,
      colors: C.SEAT_COLORS,
      wallSeats: C.wallSeats(events.map(e => Object.assign({}, e, { applied: true }))),
      lastMove,
      lastSeat: last ? last.seat : null,
      prevCell: C.cellBefore(state.events, n),
      dotSeat,
      dotSelected: selectedSeat !== null && !rv,
      legalDots: [...legalPawns],
      legalPawns,
      legalWalls,
      wallsAllowed: canWall,
      candidates: currentCandidates(),
      pv: pvOverlay,
      inactive: rv || pos.finished || !ready(),
    };
  }

  function renderBoard() {
    board.render(buildView());
  }

  function setPv(cand) {
    const next = cand && pvOnHover && cand.pv && cand.pv.length
      ? cand.pv.map(([letter, move]) => ({ seat: SEAT.indexOf(letter), move })).filter(s => s.seat >= 0) : null;
    const key = next ? next.map(s => s.seat + s.move).join() : "";
    if (key === pvKey) return;
    pvKey = key;
    pvOverlay = next;
    renderBoard();
  }

  // -- header and banner ---------------------------------------------------------------------------------
  function renderHeader() {
    let t = "Starting…";
    if (state) {
      const s = state;
      if (s.engine.status === "starting") t = "Loading the engine…";
      else if (s.engine.status !== "ready") t = "Engine stopped";
      else t = [s.mode === "analysis" ? "analysis (all human)" : "play", "opponent: " + s.settings.opponentMode,
        "default " + s.settings.maxVisits + " visits"].join(" · ");
    }
    $("engineInfo").textContent = t;
  }

  function renderBanner() {
    const b = $("banner");
    let title = "", msg = "", restart = false;
    if (lockedOut) {
      title = "Another browser tab controls this game.";
      msg = "Close it, or reload this page to take over.";
    } else if (connLost) {
      title = "Lost the connection to the server.";
      msg = connMsg + " The page reconnects on its own.";
    } else if (state && state.engine.status !== "ready" && state.engine.status !== "starting") {
      title = "The engine stopped.";
      msg = (state.engine.message || "") + "\nRestart it to continue (the game is replayed).";
      restart = true;
    }
    b.hidden = !title;
    $("bannerTitle").textContent = title;
    $("bannerMsg").textContent = msg;
    $("restartBtn").hidden = !restart;
  }

  // -- seat cards -------------------------------------------------------------------------------------------
  function buildSeatCards() {
    const box = $("seats");
    box.textContent = "";
    seatCards = [];
    for (let i = 0; i < 4; i++) {
      const card = document.createElement("div");
      card.className = "pcard";
      card.style.setProperty("--acc", C.SEAT_COLORS[i]);
      card.innerHTML =
        '<svg width="26" height="26" aria-hidden="true"><circle cx="13" cy="13" r="9.5" fill="' + C.SEAT_COLORS[i] +
        '" stroke="#fff" stroke-width="3"/><circle cx="13" cy="13" r="11" fill="none" stroke="' + C.SEAT_COLORS[i] + '" stroke-width="1.5"/></svg>' +
        '<div class="who"><span class="name"></span> <span class="sub"></span></div>' +
        '<div class="ctl"><select class="kind" aria-label="player"><option value="human">Human</option><option value="net">AI net</option><option value="bot">Bot</option></select>' +
        '<button type="button" class="elim" title="Eliminate this seat (its walls stay on the board; undo brings it back)">✕</button></div>' +
        '<div class="row2"><span class="sub stats"></span><div class="walls"></div></div>' +
        '<div class="row3" hidden><input type="number" min="0" placeholder="visits (default)" aria-label="visits" hidden>' +
        '<select class="bot" aria-label="bot" hidden></select><select class="target" aria-label="grudge target" hidden></select></div>';
      box.appendChild(card);
      const cell = { el: card, i };
      cell.name = card.querySelector(".name");
      cell.sub = card.querySelector(".who .sub");
      cell.stats = card.querySelector(".stats");
      cell.walls = card.querySelector(".walls");
      cell.kindSel = card.querySelector("select.kind");
      cell.row3 = card.querySelector(".row3");
      cell.visits = card.querySelector('input[type="number"]');
      cell.botSel = card.querySelector("select.bot");
      cell.targetSel = card.querySelector("select.target");
      cell.elim = card.querySelector(".elim");
      cell.botSel.innerHTML = BOTS.map(b => '<option value="' + b + '">' + b + "</option>").join("");
      cell.targetSel.innerHTML = '<option value="">target…</option>' + SEAT.map((l, k) =>
        '<option value="' + k + '">' + l + "</option>").join("");
      for (const c of [cell.kindSel, cell.visits, cell.botSel, cell.targetSel]) c.addEventListener("change", () => sendSeats());
      cell.elim.addEventListener("click", () => eliminate(i));
      seatCards.push(cell);
    }
  }

  function renderSeats() {
    const s = state, pos = shownPosition(), rv = reviewing();
    const initial = s.rules.initialWalls;
    seatCards.forEach((cell) => {
      const i = cell.i, seat = s.seats[i];
      const isAlive = pos.alive[i];
      cell.el.classList.toggle("tomove", !pos.finished && pos.toMove === i);
      cell.el.classList.toggle("dead", !isAlive);
      cell.name.textContent = SEAT[i];
      cell.sub.textContent = C.SEAT_NAMES[i] + (isAlive ? "" : " · eliminated");
      const d = pos.dist[i];
      const left = pos.wallsLeft[i];
      cell.stats.textContent = !isAlive ? "out · walls lost"
        : (d === null || d === undefined ? "? steps" : d >= 255 ? "no path" : d + " step" + (d === 1 ? "" : "s")) + " to go · " + left + " wall" + (left === 1 ? "" : "s");
      cell.walls.title = left + " walls left";
      let pips = "";
      for (let k = 0; k < Math.min(10, initial[i]); k++) pips += '<i class="' + (k < left ? "" : "used") + '"></i>';
      cell.walls.innerHTML = pips;
      setIfIdle(cell.kindSel, seat.kind);
      cell.visits.hidden = seat.kind !== "net";
      cell.botSel.hidden = seat.kind !== "bot";
      cell.targetSel.hidden = !(seat.kind === "bot" && seat.bot === "grudge");
      cell.row3.hidden = seat.kind === "human";
      setIfIdle(cell.visits, seat.visits ? String(seat.visits) : "", "visits (default)");
      setIfIdle(cell.botSel, seat.bot);
      setIfIdle(cell.targetSel, seat.target === null || seat.target === undefined ? "" : String(seat.target));
      cell.elim.disabled = !isAlive || pos.finished || busy || rv || !ready();
    });
  }

  function setIfIdle(el, value, placeholder) {
    if (document.activeElement === el) return;
    if (el.tagName === "INPUT") {
      el.value = value;
      if (placeholder) el.placeholder = placeholder;
    } else if (el.value !== value) {
      el.value = value;
    }
  }

  function collectSeats() {
    return seatCards.map((cell) => {
      const kind = cell.kindSel.value;
      const seat = { kind };
      if (kind === "net") seat.visits = cell.visits.value ? parseInt(cell.visits.value, 10) : 0;
      if (kind === "bot") {
        seat.bot = cell.botSel.value;
        if (seat.bot === "grudge") seat.target = cell.targetSel.value === "" ? null : parseInt(cell.targetSel.value, 10);
      }
      return seat;
    });
  }

  function sendSeats(me) {
    const body = { seats: collectSeats() };
    if (me !== undefined) body.me = me;
    return act("Setting the seats…", () => api("POST", "/api/seats", body));
  }

  async function eliminate(seat) {
    const alive = state.position.alive.filter(Boolean).length;
    let text = "Its walls stay on the board; its unplaced walls are lost. Undo brings it back.";
    const last = alive === 2;
    if (last) text += " This leaves ONE seat: the game ends at once (undo reopens it).";
    if (!(await askConfirm("Eliminate " + SEAT[seat] + " (" + C.SEAT_NAMES[seat] + ")?", text, "Eliminate"))) return;
    viewPly = null;
    await act("Eliminating…", () => api("POST", "/api/eliminate", { seat, confirm: last }));
  }

  // -- evaluation ---------------------------------------------------------------------------------------------
  const pct = (p, d) => C.pct(p, d);
  function listText(xs, d, sign) {
    if (!xs) return "–";
    return xs.map((x, k) => SEAT[k] + " " + (sign && x >= 0 ? "+" : "") + x.toFixed(d)).join("  ");
  }

  function leadText(p, pos) {
    if (pos.finished) return pos.winner >= 0 ? SEAT[pos.winner] + " (" + C.SEAT_NAMES[pos.winner] + ") wins" : "Draw";
    const order = [0, 1, 2, 3].sort((a, b) => p[b] - p[a]);
    if (p[order[0]] - p[order[1]] < 0.03) return "Even race";
    return SEAT[order[0]] + " leads · " + pct(p[order[0]]);
  }

  function renderEval() {
    const n = curPly(), pos = shownPosition(), an = state.analysis;
    $("evalPly").textContent = n === 0 ? "start position" : "after event " + n;
    let p = null, src = "";
    if (!reviewing() && an.root && an.root.value && an.ply === state.cursor) { p = an.root.value; src = "Search · " + an.root.visits + " visits"; }
    else if (state.evals[n]) { p = state.evals[n]; src = "Stored evaluation"; }
    else if (!reviewing() && pos.raw && pos.raw.value) { p = pos.raw.value; src = "Net estimate (no search)"; }
    const bar = $("evBar"), nums = $("evNums");
    bar.textContent = ""; nums.textContent = "";
    const cols = C.SEAT_COLORS.concat([C.DRAW_COLOR]);
    if (p) {
      C.barSegments(p).forEach((f, k) => {
        const d = document.createElement("div");
        d.style.width = (100 * f) + "%";
        d.style.background = cols[k];
        bar.appendChild(d);
      });
      p.forEach((x, k) => {
        const it = document.createElement("span");
        const sw = document.createElement("i");
        sw.style.background = cols[k];
        it.appendChild(sw);
        it.appendChild(document.createTextNode((k < 4 ? SEAT[k] : "draw") + " " + pct(x)));
        nums.appendChild(it);
      });
      $("evLead").textContent = leadText(p, pos);
    } else {
      $("evLead").textContent = an.on && !reviewing() ? "Evaluating…" : "No evaluation";
    }
    $("evSrc").textContent = src;
    const raw = reviewing() ? null : pos.raw;
    $("evNet").textContent = raw && raw.value && !src.startsWith("Net")
      ? "Net alone: " + SEAT.map((l, k) => l + " " + pct(raw.value[k], 0)).join(" ") + " draw " + pct(raw.value[4], 0) +
        (raw.plies ? " · plies to end ≈ " + Math.round(raw.plies) : "")
      : "";
    renderChart(n);
  }

  function renderChart(n) {
    const svg = $("chart");
    const W = Math.max(200, svg.clientWidth || 300), H = 84, L = 22, R = 6, T = 6, B = 6;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.textContent = "";
    const len = state.length, span = Math.max(len, 10);
    const X = i => L + (i / span) * (W - L - R), Y = v => T + (1 - v) * (H - T - B);
    svgEl("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "var(--chart-bg)", rx: 4 }, svg);
    svgEl("line", { x1: L, x2: W - R, y1: Y(0.25), y2: Y(0.25), stroke: "var(--line)", "stroke-dasharray": "3 3" }, svg);
    svgEl("line", { x1: L, x2: W - R, y1: Y(0.5), y2: Y(0.5), stroke: "var(--line)", "stroke-dasharray": "3 3" }, svg);
    const lab = (y, s) => { const t = svgEl("text", { x: L - 4, y, "text-anchor": "end", "font-size": 9.5, fill: "var(--muted)" }, svg); t.textContent = s; };
    lab(T + 8, "100"); lab(Y(0.5) + 3, "50"); lab(H - B - 1, "0");
    for (let s = 0; s < 4; s++) {
      const pts = [];
      state.evals.forEach((p, i) => { if (p) pts.push([X(i), Y(p[s])]); });
      if (!pts.length) continue;
      svgEl("path", { d: pts.map((q, k) => (k ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join(""), fill: "none",
        stroke: C.SEAT_COLORS[s], "stroke-width": 1.8, "stroke-linejoin": "round" }, svg);
    }
    svgEl("line", { x1: X(n), x2: X(n), y1: T, y2: H - B, stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0.7 }, svg);
    svg.onclick = (ev) => {
      const r = svg.getBoundingClientRect();
      const x = (ev.clientX - r.left) * (W / r.width);
      navigate(Math.max(0, Math.min(len, Math.round(((x - L) / (W - L - R)) * span))));
    };
  }

  // -- candidates ---------------------------------------------------------------------------------------------
  function renderCands() {
    const card = $("candCard");
    const cands = currentCandidates();
    if (!cands.length) { card.hidden = true; candKey = ""; return; }
    card.hidden = false;
    const pos = state.position, an = state.analysis, r = an.root, mover = pos.toMove;
    $("candTitle").textContent = "Analysis";
    $("candSub").textContent = SEAT[mover] + " to move" + (r ? " · " + U.fmtVisits(r.visits) + " visits" : "");
    let root = "";
    if (r && r.value) {
      root += "<div><b>" + SEAT[mover] + " " + pct(r.value[mover]) + "</b> · utility " + esc(listText(r.utility, 3, true)) + "</div>";
      root += '<div class="muted">lead (cells) ' + esc(listText(r.lead, 2, true)) + "</div>";
    }
    if (pos.raw && pos.raw.value) root += '<div class="muted">net value: ' + SEAT[mover] + " " + pct(pos.raw.value[mover]) + "</div>";
    $("candRoot").innerHTML = root;

    const rows = C.sortCandidates(cands.map(c => Object.assign({}, c)), sortKey);
    const key = [sortKey, mover].concat(rows.map(c => c.move + ":" + c.visits)).join(",");
    if (key === candKey) return;
    candKey = key;
    const top = Math.max(...cands.map(c => c.visits));
    const total = an.moves.reduce((t, m) => t + (m.visits > 0 ? m.visits : 0), 0) || 1;
    const sh = (k, label, title) => '<span class="sort' + (sortKey === k ? " on" : "") + '" data-sort="' + k + '" title="' + title + ' · click to sort">' + label + (sortKey === k ? " ▾" : "") + "</span>";
    const head = '<div class="crow chead"><span title="Rank by lcb">#</span><span>move</span>' +
      sh("win", "win%", "Win chance for " + SEAT[mover] + " (the seat to move)") +
      sh("visits", "visits", "Visits (share of all child visits)") +
      sh("prior", "policy", "Raw policy prior of the net") +
      sh("utility", "utility", "Utility for " + SEAT[mover]) +
      sh("lcb", "lcb", "Lower confidence bound") + "</div>";
    const colors = C.SEAT_COLORS.concat([C.DRAW_COLOR]);
    $("candList").innerHTML = head + rows.map((m) => {
      const rank = cands.indexOf(cands.find(c => c.move === m.move));
      const best = rank === 0;
      let stack = "";
      if (m.value) {
        stack = '<span class="stack" title="after the move: ' + esc(m.value.map((x, k) => (k < 4 ? SEAT[k] : "draw") + " " + pct(x, 0)).join("  ")) + '">' +
          C.barSegments(m.value).map((f, k) => '<div style="width:' + (100 * f) + "%;background:" + colors[k] + '"></div>').join("") + "</span>";
      }
      const pvs = (m.pv || []).slice(0, 10).filter(s => SEAT.includes(s[0]) && C.parseMove(s[1])).map(s => s[0] + " " + s[1]).join(" ");
      const alpha = best ? 1 : U.candAlpha(m.visits, top);
      return '<div class="crow' + (best ? " top" : "") + '" data-move="' + esc(m.move) + '" style="--dot:' + (best ? U.CAND_TOP : U.CAND_OFF) + ";--dot-a:" + alpha + '">' +
        '<span class="ci"><i></i>' + (rank + 1) + "</span><span class=\"mv\">" + esc(m.move) + "</span>" +
        "<span>" + pct(m.win, 1).replace("%", "") + "</span>" +
        "<span>" + U.fmtVisits(m.visits) + " <small>" + (100 * m.visits / total).toFixed(0) + "%</small></span>" +
        "<span>" + (m.prior == null ? "–" : (100 * m.prior).toFixed(m.prior < 0.001 ? 3 : 1) + "%") + "</span>" +
        "<span>" + (m.utility == null ? "–" : (m.utility >= 0 ? "+" : "−") + Math.abs(m.utility).toFixed(2)) + "</span>" +
        "<span>" + (m.lcb == null ? "–" : m.lcb.toFixed(3)) + "</span>" +
        '<span class="cpv">' + stack + esc(pvs) + "</span></div>";
    }).join("") + '<div class="note">Values for ' + SEAT[mover] + ", the seat to move. Hover a line to see its continuation; click to play it.</div>";
  }

  // -- move list ----------------------------------------------------------------------------------------------
  function renderMoves() {
    const n = curPly();
    const key = JSON.stringify([state.events.map(e => [e.seat, e.a || "", !!e.elim, !!e.ai, e.applied]), state.evals.map(p => p ? Math.round(1000 * p[0]) + "," + Math.round(1000 * p[1]) + "," + Math.round(1000 * p[2]) + "," + Math.round(1000 * p[3]) : 0), showEval]);
    if (key !== movesKey) {
      movesKey = key;
      const rows = ['<div class="m" data-ply="0"><span class="n"></span><span class="who"></span><span class="mv muted">start</span><span class="ev"></span></div>'];
      let num = 0;
      state.events.forEach((e, k) => {
        const p = state.evals[k];
        const cls = "m" + (e.applied ? "" : " redo") + (e.elim ? " elim" : "");
        const mv = e.elim ? "EL " + SEAT[e.seat] : e.a;
        if (!e.elim) num += 1;
        rows.push('<div class="' + cls + '" data-ply="' + (k + 1) + '"' + (e.applied ? "" : ' title="Undone: click to go back to it"') + '><span class="n">' + (e.elim ? "" : num) +
          '</span><span class="who s' + e.seat + '"></span><span class="mv' + (e.elim ? "" : (e.a || "").length > 3 && /[hv]$/.test(e.a) ? " wall" : "") + '">' + esc(mv) +
          (e.ai ? '<span class="tag">AI</span>' : "") + '</span><span class="ev">' + (p && !e.elim ? SEAT[e.seat] + " " + Math.round(100 * p[e.seat]) + "%" : "") + "</span></div>");
      });
      $("moveList").innerHTML = rows.join("");
    }
    $("moveCount").textContent = state.length ? state.cursor + " of " + state.length + " events" + (state.cursor < state.length ? " · " + (state.length - state.cursor) + " undone" : "") : "";
    document.querySelectorAll("#moveList .m.cur").forEach(x => x.classList.remove("cur"));
    const row = document.querySelector('#moveList .m[data-ply="' + n + '"]');
    if (row) {
      row.classList.add("cur");
      // scroll only the list (scrollIntoView would also scroll the page)
      const list = $("moveList");
      if (row.offsetTop < list.scrollTop) list.scrollTop = row.offsetTop;
      else if (row.offsetTop + row.offsetHeight > list.scrollTop + list.clientHeight) list.scrollTop = row.offsetTop + row.offsetHeight - list.clientHeight;
    }
    $("reviewBar").hidden = !reviewing();
    $("reviewText").textContent = reviewing() ? "Position " + viewPly + " of " + state.length + " (read-only)" : "";
  }

  // -- status, overlay, buttons ---------------------------------------------------------------------------------
  function renderStatus() {
    const st = $("status");
    let html = "";
    const pos = state && state.position;
    if (!state || state.engine.status === "starting") html = '<span class="spin"></span>Loading the engine and the net… (the first start can take a while)';
    else if (state.engine.status !== "ready") html = "The engine stopped: use <b>Restart engine</b> above.";
    else if (reviewing()) html = "Reviewing the position after event " + viewPly + " of " + state.length + " (read-only). <a href=\"#\" id=\"stBack\">Back to current</a>";
    else if (busy) html = '<span class="spin"></span>' + esc(busyText);
    else if (aiRunning) html = '<span class="spin"></span>The AI seats are playing…';
    else if (pos.finished) html = "<b>Game over:</b> " + (pos.winner >= 0 ? SEAT[pos.winner] + " (" + C.SEAT_NAMES[pos.winner] + ") wins." : "draw.") +
      (state.cursor > 0 ? " ← to step back." : "");
    else {
      const seat = state.seats[pos.toMove], tm = SEAT[pos.toMove];
      const an = state.analysis;
      const ana = an.running ? ' · <span class="spin"></span>analyzing ' + U.fmtVisits(an.root ? an.root.visits : 0) + " / " + (an.cap ? U.fmtVisits(an.cap) : "∞") + " visits · Space pauses"
        : an.on ? " · analysis " + (an.done ? "done" : "paused") : " · Space starts the analysis";
      if (seat.kind === "human") html = "<b>" + tm + " to move</b> (" + C.SEAT_NAMES[pos.toMove] + "): click a dot to move the pawn" +
        (pos.wallsLeft[pos.toMove] > 0 ? ", or a groove between cells to place a wall." : ". No walls left.") + ana;
      else html = "<b>" + tm + " to move</b> (" + esc(kindLabel(seat)) + "): press <b>AI move</b>, or play for it." + ana;
    }
    st.innerHTML = html;
    const back = $("stBack");
    if (back) back.onclick = (e) => { e.preventDefault(); navigate(null); };
    $("moveInput").disabled = !(state && canPlay());
  }

  function renderOverlay() {
    const pos = state.position;
    const key = state.cursor + ":" + state.length;
    const show = pos.finished && !reviewing() && dismissedOver !== key && !isAnalysis();
    $("overlay").hidden = !show;
    if (!show) return;
    $("ovTitle").textContent = pos.winner >= 0 ? SEAT[pos.winner] + " (" + C.SEAT_NAMES[pos.winner] + ") wins" : "Draw";
    $("ovSub").textContent = pos.winner >= 0 ? (state.seats[pos.winner].kind === "human" ? "Well played!" : "The " + kindLabel(state.seats[pos.winner]) + " seat wins this one.") : "The game is drawn.";
    $("overlay").style.setProperty("--acc", pos.winner >= 0 ? C.SEAT_COLORS[pos.winner] : "var(--board)");
  }

  function renderButtons() {
    const rd = ready(), idle = rd && !busy && !aiRunning;
    const cur = state ? state.cursor : 0, len = state ? state.length : 0;
    $("undoBtn").disabled = !(idle && cur > 0);
    $("redoBtn").disabled = !(idle && cur < len);
    $("newBtn").disabled = !rd || busy;
    $("saveBtn").disabled = !idle;
    $("loadBtn").disabled = !idle;
    $("anaBtn").disabled = !(rd && !busy && !state.position.finished);
    const on = !!state && state.analysis.on;
    $("anaBtn").classList.toggle("on", on);
    $("anaBtn").textContent = on ? "❚❚ Analysis" : "▶ Analysis";
    $("aiNowBtn").disabled = !(idle && isAiToMove() && !reviewing());
    $("aiAllBtn").disabled = !(idle && isAiToMove() && !reviewing());
    $("aiStopBtn").hidden = !aiRunning;
    $("redoBtn").title = "Replay the next event (→ in analysis mode)";
    const mv = $("maxVisits");
    if (state && document.activeElement !== mv) mv.value = state.analysis.cap ? String(state.analysis.cap) : "";
    if (state && document.activeElement !== $("anaCs")) $("anaCs").value = state.analysis.cs;
    $("evalBtn").classList.toggle("on", showEval);
    $("pvBtn").classList.toggle("on", pvOnHover);
    document.body.classList.toggle("noeval", !showEval);
    $("keys").textContent = isAnalysis()
      ? "Space analysis · ← → / wheel on the board undo / redo · Home End · U undo · R rotate · P PV on hover · E evaluation · N new"
      : "Space analysis · ← → / wheel review (read-only) · Home End · U undo · R rotate · P PV on hover · E evaluation · N new · Esc back";
    document.querySelectorAll("#seatView button").forEach((b, k) => b.classList.toggle("on", k === rot));
  }

  function fillSettings() {
    const st = state.settings;
    $("setOpp").value = st.opponentMode;
    $("setMe").value = state.me === null || state.me === undefined ? "" : String(state.me);
    $("setVisits").value = st.maxVisits;
    $("setLead").value = st.dynamicScoreUtilityFactor;
    $("setEps").value = st.q4LeadWeightEps;
    $("setCpuct").value = st.cpuctExploration;
    $("setRpt").value = st.rootPolicyTemperature;
    settingsFilled = true;
  }
  let settingsFilled = false;

  // -- top-level render -----------------------------------------------------------------------------------
  function render() {
    renderBanner();
    renderHeader();
    if (!state) { renderStatus(); renderButtons(); return; }
    renderBoard();
    renderSeats();
    renderEval();
    renderCands();
    renderMoves();
    renderOverlay();
    renderStatus();
    renderButtons();
    if (!settingsFilled) fillSettings();
  }

  // -- actions -----------------------------------------------------------------------------------------------
  function playMove(text) {
    pendingLegal = null;
    selectedSeat = null;
    viewPly = null;
    board.hover = null;
    setPv(null);
    return act("Playing " + text + "…", () => api("POST", "/api/play", { move: text }));
  }

  // Going to a position: in analysis mode through the engine (undo / redo are engine moves, one event each), in a
  // game against the AI as a read-only review of the event list (the Undo button still goes through the engine).
  function navigate(ply) {
    if (!state) return;
    if (isAnalysis()) {
      viewPly = null;
      const target = ply === null ? state.length : Math.max(0, Math.min(state.length, ply));
      if (target === state.cursor || busy) return;
      selectedSeat = null; pendingLegal = null; setPv(null);
      return act("Moving to the position…", () => api("POST", "/api/goto", { ply: target }));
    }
    const target = ply === null ? state.cursor : Math.max(0, Math.min(state.length, ply));
    viewPly = target === state.cursor ? null : target;
    selectedSeat = null; pendingLegal = null; setPv(null);
    render();
  }

  // one event back / forward (wheel, arrow keys): the engine in analysis mode, the read-only review otherwise
  function step(dir) {
    if (!state || busy || !ready()) return;
    if (isAnalysis()) {
      if (dir < 0 && state.cursor > 0) undo();
      else if (dir > 0 && state.cursor < state.length) redo();
    } else {
      navigate(curPly() + dir);
    }
  }

  function undo() { viewPly = null; selectedSeat = null; pendingLegal = null; setPv(null); return act("Undo…", () => api("POST", "/api/undo")); }
  function redo() { viewPly = null; selectedSeat = null; pendingLegal = null; setPv(null); return act("Redo…", () => api("POST", "/api/redo")); }

  async function aiNow() {
    viewPly = null;
    return act("The AI is thinking…", () => api("POST", "/api/ai", { all: false }));
  }

  async function aiUntilHuman() {
    if (aiRunning) return;
    aiRunning = true;
    viewPly = null;
    render();
    try {
      for (;;) {
        let r;
        try {
          r = await api("POST", "/api/ai", { all: true });
        } catch (e) {
          toast(e.message, true);
          break;
        }
        await refresh();
        if (!r.more || !aiRunning) break;
      }
    } finally {
      aiRunning = false;
      await refresh().catch(() => {});
      render();
    }
  }

  function toggleAnalysis() {
    if (!state || busy || !ready()) return;
    return act(state.analysis.on ? "Pausing the analysis…" : "Starting the analysis…",
      () => api("POST", "/api/analysis", { on: !state.analysis.on }));
  }

  function applyCap() {
    const text = $("maxVisits").value.trim();
    const cap = text === "" ? 0 : parseInt(text, 10);
    if (!(cap >= 0)) { toast("max visits: empty (no limit) or a number ≥ 1", true); return; }
    return act("Applying…", () => api("POST", "/api/analysis", { cap }));
  }

  function applySettings() {
    const body = {
      opponentMode: $("setOpp").value,
      maxVisits: parseInt($("setVisits").value, 10),
      dynamicScoreUtilityFactor: parseFloat($("setLead").value),
      q4LeadWeightEps: parseFloat($("setEps").value),
      cpuctExploration: parseFloat($("setCpuct").value),
      rootPolicyTemperature: parseFloat($("setRpt").value),
    };
    const cs = parseInt($("anaCs").value, 10);
    if (!(cs >= 1)) { toast("interval must be a number of centiseconds (≥ 1)", true); return; }
    const me = $("setMe").value === "" ? null : parseInt($("setMe").value, 10);
    const same = state.me === me || (state.me == null && me == null);
    return act("Applying the settings…", async () => {
      await api("POST", "/api/settings", body);
      if (cs !== state.analysis.cs) await api("POST", "/api/analysis", { cs });
      if (!same) await api("POST", "/api/seats", { seats: collectSeats(), me });
    }).then((r) => { settingsFilled = false; fillSettingsLater(); return r; });
  }
  function fillSettingsLater() { if (state) fillSettings(); }

  async function showParams() {
    try {
      const r = await api("GET", "/api/params");
      $("paramsOut").textContent = r.text;
      $("paramsOut").hidden = false;
    } catch (e) {
      toast(e.message, true);
    }
  }

  function submitMove() {
    const r = C.normaliseMoveInput($("moveInput").value);
    if (r.error) { toast(r.error, true); return; }
    if (!canPlay()) { toast("The game is not at a position you can move in.", true); return; }
    if (!(state.position.legal || []).includes(r.move)) { toast(r.move + " is not legal here", true); return; }
    $("moveInput").value = "";
    playMove(r.move);
  }

  async function saveSgf() {
    try {
      const r = await fetch("/api/sgf", { headers: { "X-Client": CLIENT } });
      if (!r.ok) throw failure((await r.json()).error, r.status);
      const blob = await r.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "kataquoridor-q4.sgfs";
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
      toast("Saved kataquoridor-q4.sgfs");
    } catch (e) {
      toast("SGF: " + e.message, true);
    }
  }

  function loadGame() {
    const body = { text: $("loadText").value };
    const plyText = $("loadPly").value.trim();
    if (plyText !== "") body.ply = parseInt(plyText, 10);
    viewPly = null; selectedSeat = null; pendingLegal = null;
    return act("Loading…", () => api("POST", "/api/load", body));
  }

  async function restart() {
    await act("Restarting the engine…", () => api("POST", "/api/restart"));
  }

  function rotateTo(r) {
    rot = ((r % 4) + 4) % 4;
    board.setRotation(rot);
    document.querySelectorAll("#seatView button").forEach((b, k) => b.classList.toggle("on", k === rot));
  }

  // A pawn of another seat was clicked: its legal moves (a read from the engine), shown as dots.
  async function clickCell(name) {
    if (!state || reviewing()) return;
    const pos = state.position;
    const seat = [0, 1, 2, 3].find(s => pos.pawns[s] === name && pos.alive[s]);
    if (seat === undefined) {
      if (selectedSeat !== null) { selectedSeat = null; pendingLegal = null; renderBoard(); }
      return;
    }
    selectedSeat = selectedSeat === seat ? null : seat;
    if (selectedSeat === null || seat === pos.toMove) { pendingLegal = null; renderBoard(); return; }
    try {
      const r = await api("GET", "/api/legal?seat=" + seat);
      pendingLegal = { seat, legal: r.legal };
    } catch (e) {
      toast(e.message, true);
      pendingLegal = null;
    }
    renderBoard();
  }

  // -- new game dialog ----------------------------------------------------------------------------------------
  function visitsOf(st) {
    if (st.preset === "custom") return Math.max(1, Math.min(100000, parseInt(st.custom, 10) || 400));
    return PRESETS[st.preset] || PRESETS.normal;
  }

  function openNewDialog() {
    if (!state || !ready() || busy) return;
    const st = loadSettings();
    const side = st.side || "keep", preset = st.preset || "hard";
    const dlg = $("newDlg");
    dlg.querySelectorAll('input[name="side"]').forEach(i => { i.checked = i.value === side; });
    dlg.querySelectorAll('input[name="preset"]').forEach(i => { i.checked = i.value === preset; });
    if (st.custom) $("customVisits").value = st.custom;
    $("newPlies").value = state.rules.maxPlies;
    $("newRep").value = state.rules.repetitionDrawCount;
    state.rules.initialWalls.forEach((w, k) => { $("newW" + k).value = w; });
    $("newWarn").hidden = state.length === 0;
    syncNewDialog();
    dlg.showModal();
    $("startBtn").focus();
  }

  function syncNewDialog() {
    const side = ($("newDlg").querySelector('input[name="side"]:checked') || {}).value;
    $("strField").classList.toggle("off", side === "keep" || side === "analysis");
  }

  function startNewGame() {
    const dlg = $("newDlg");
    const side = (dlg.querySelector('input[name="side"]:checked') || {}).value || "keep";
    const preset = (dlg.querySelector('input[name="preset"]:checked') || {}).value || "hard";
    const custom = parseInt($("customVisits").value, 10) || 400;
    saveSettings({ side, preset, custom });
    const visits = visitsOf({ preset, custom });
    let seats, human = null;
    if (side === "keep") seats = collectSeats();
    else if (side === "analysis") seats = [0, 1, 2, 3].map(() => ({ kind: "human" }));
    else if (side === "none") seats = [0, 1, 2, 3].map(() => ({ kind: "net", visits }));
    else {
      human = side === "random" ? Math.floor(Math.random() * 4) : parseInt(side, 10);
      seats = [0, 1, 2, 3].map(i => (i === human ? { kind: "human" } : { kind: "net", visits }));
    }
    const rules = {
      maxPlies: parseInt($("newPlies").value, 10),
      repetitionDrawCount: parseInt($("newRep").value, 10),
      initialWalls: [0, 1, 2, 3].map(k => parseInt($("newW" + k).value, 10)),
    };
    selectedSeat = null; pendingLegal = null; viewPly = null; dismissedOver = null; setPv(null);
    if (human !== null) rotateTo(human);
    return act("Starting a new game…", () => api("POST", "/api/new", { rules, seats }));
  }

  // -- wiring ---------------------------------------------------------------------------------------------------
  function wire() {
    board = new Q4Board($("board"), {
      onMove: (name) => playMove(name),
      onCell: (name) => clickCell(name),
      onIllegal: (name) => toast(name + " is not a legal wall here", true),
      onHover: (h) => {
        const c = h && currentCandidates().find(x => x.move === h.name);
        setPv(c || null);
      },
    });
    board.setRotation(0);
    const seatView = $("seatView");
    SEAT.forEach((letter, k) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = letter;
      b.title = "put " + C.SEAT_NAMES[k] + " at the bottom";
      b.addEventListener("click", () => rotateTo(k));
      seatView.appendChild(b);
    });
    rotateTo(0);
    buildSeatCards();

    $("rotateBtn").addEventListener("click", () => rotateTo(rot + 1));
    $("moveForm").addEventListener("submit", (e) => { e.preventDefault(); submitMove(); });
    $("undoBtn").addEventListener("click", undo);
    $("redoBtn").addEventListener("click", redo);
    $("aiNowBtn").addEventListener("click", aiNow);
    $("aiAllBtn").addEventListener("click", aiUntilHuman);
    $("aiStopBtn").addEventListener("click", () => { aiRunning = false; });
    $("anaBtn").addEventListener("click", toggleAnalysis);
    $("maxVisits").addEventListener("change", applyCap);
    $("setApply").addEventListener("click", applySettings);
    $("paramsBtn").addEventListener("click", showParams);
    $("saveBtn").addEventListener("click", saveSgf);
    $("loadBtn").addEventListener("click", () => $("loadDlg").showModal());
    $("newBtn").addEventListener("click", openNewDialog);
    $("restartBtn").addEventListener("click", restart);
    $("backBtn").addEventListener("click", () => navigate(null));
    $("ovNew").addEventListener("click", openNewDialog);
    $("ovReview").addEventListener("click", () => { dismissedOver = state.cursor + ":" + state.length; render(); });
    $("pvBtn").addEventListener("click", () => { pvOnHover = !pvOnHover; saveSettings({ pvOnHover }); if (!pvOnHover) setPv(null); render(); });
    $("evalBtn").addEventListener("click", () => { showEval = !showEval; saveSettings({ showEval }); movesKey = ""; render(); });

    // new game / load dialogs
    $("customVisits").addEventListener("focus", () => { $("newDlg").querySelector('input[value="custom"]').checked = true; });
    $("newDlg").querySelectorAll('input[name="side"]').forEach(i => i.addEventListener("change", syncNewDialog));
    $("newDlg").addEventListener("close", () => { if ($("newDlg").returnValue === "ok") startNewGame(); });
    $("loadDlg").addEventListener("close", () => { if ($("loadDlg").returnValue === "ok") loadGame(); });
    $("loadFile").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => { $("loadText").value = String(reader.result).split(/\r?\n/).find(l => l.trim()) || ""; };
      reader.readAsText(f);
    });

    // candidate table: pointerdown, not click (the list is rebuilt while the analysis runs, which can swallow a click)
    const list = $("candList");
    list.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      const sc = e.target.closest("[data-sort]");
      if (sc) { sortKey = sc.dataset.sort; saveSettings({ sortKey }); candKey = ""; renderCands(); return; }
      const row = e.target.closest(".crow");
      if (row && row.dataset.move && canPlay() && (state.position.legal || []).includes(row.dataset.move)) playMove(row.dataset.move);
    });
    list.addEventListener("mouseover", (e) => {
      const row = e.target.closest(".crow");
      setPv(row && row.dataset.move ? currentCandidates().find(c => c.move === row.dataset.move) : null);
    });
    list.addEventListener("mouseleave", () => setPv(null));

    // move list: click to go to a position; the wheel over it scrolls the list as usual
    $("moveList").addEventListener("click", (e) => {
      const row = e.target.closest(".m");
      if (row) navigate(+row.dataset.ply);
    });

    // mouse wheel over the board: up = one event back (undo), down = one event forward (redo). Trackpads send many
    // small deltas, so they are summed; wheel events that arrive while the engine is still answering are dropped.
    let wheelAcc = 0, wheelTimer = null;
    $("board").addEventListener("wheel", (e) => {
      if (!state) return;
      e.preventDefault();
      if (busy || aiRunning) { wheelAcc = 0; return; }
      wheelAcc += e.deltaMode === 1 ? 40 * e.deltaY : e.deltaY;
      clearTimeout(wheelTimer);
      wheelTimer = setTimeout(() => { wheelAcc = 0; }, 300);
      if (Math.abs(wheelAcc) < 40) return;
      const dir = wheelAcc > 0 ? 1 : -1;
      wheelAcc = 0;
      step(dir);
    }, { passive: false });

    document.addEventListener("keydown", (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey || document.querySelector("dialog[open]")) return;
      const tag = (e.target && e.target.tagName) || "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) {
        if (e.key === "Escape" || (e.key === "Enter" && e.target.id === "maxVisits")) e.target.blur();
        return;
      }
      if (!state) return;
      const k = e.key;
      if (k === " ") { e.preventDefault(); if (e.target.blur) e.target.blur(); toggleAnalysis(); }
      else if (k === "n" || k === "N") { e.preventDefault(); openNewDialog(); }
      else if (k === "u" || k === "U" || k === "Backspace") { e.preventDefault(); if (!$("undoBtn").disabled) undo(); }
      else if (k === "r" || k === "R") rotateTo(rot + 1);
      else if (k === "e" || k === "E") $("evalBtn").click();
      else if (k === "p" || k === "P") $("pvBtn").click();
      else if (k === "ArrowLeft") { e.preventDefault(); step(-1); }
      else if (k === "ArrowRight") { e.preventDefault(); step(1); }
      else if (k === "Home") { e.preventDefault(); navigate(0); }
      else if (k === "End") { e.preventDefault(); navigate(null); }
      else if (k === "Escape") {
        if (selectedSeat !== null) { selectedSeat = null; pendingLegal = null; renderBoard(); }
        else if (reviewing()) navigate(null);
      }
    });
    window.addEventListener("resize", () => { if (state) renderChart(curPly()); });
  }

  wire();
  render();
  poll();
})();
