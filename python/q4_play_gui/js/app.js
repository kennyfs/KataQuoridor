/* The page of the Q4 play GUI: polls /api/state, draws it, sends the user's actions. The engine decides every rule;
 * this file only shows the answers. Untrusted text (engine answers, pasted records) goes in with textContent. */
(function () {
  const C = window.Q4Core;
  const Q4Board = window.Q4Board;
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
  const KINDS = [["human", "Human"], ["net", "AI net"], ["bot", "Bot"]];
  const BOTS = ["greedy", "basher", "randomPawn", "random", "grudge"];
  // The candidates shown on the board and in the table: at most SHOWN_CANDIDATES searched moves whose visits are more
  // than MIN_VISIT_SHARE of all searched children's visits, best LCB first (see shownCandidates).
  const SHOWN_CANDIDATES = 5;
  const MIN_VISIT_SHARE = 0.01;

  let state = null;
  let rot = 0;
  let busy = false;
  let selectedSeat = null;          // a pawn clicked on the board (any seat)
  let pendingLegal = null;          // {seat, legal} for a seat other than the one to move
  let pvOverlay = null;             // the hovered candidate's principal variation
  let sortKey = "lcb";
  let settingsFilled = false;
  let lastCursorShown = -1;
  let aiRunning = false;
  let lockedOut = false;
  let board = null;
  let seatCards = [];

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

  async function refresh() {
    const s = await api("GET", "/api/state");
    state = s;
    lockedOut = false;
    render();
    return s;
  }

  // Runs one action: busy flag, toast on error, state refreshed afterwards.
  async function act(text, fn) {
    if (busy) return undefined;
    busy = true;
    showBusy(text);
    try {
      const r = await fn();
      await refresh();
      return r;
    } catch (e) {
      toast(e.message, true);
      if (e.status === 423) showLockout();
      else if (e.status === 0 || e.engineDown) showBanner(e.message);
      try { await refresh(); } catch (e2) { /* the server may be gone: the banner says so */ }
      return undefined;
    } finally {
      busy = false;
      hideBusy();
    }
  }

  function showBusy(text) {
    $("busyText").textContent = text;
    document.body.style.cursor = "progress";
  }

  function hideBusy() {
    document.body.style.cursor = "";
  }

  // -- messages -------------------------------------------------------------------------------------------
  let toastTimer = null;
  function toast(text, isErr) {
    const t = $("toast");
    t.textContent = text;
    t.classList.toggle("err", !!isErr);
    t.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { t.hidden = true; }, isErr ? 6000 : 2500);
  }

  function showBanner(text) {
    $("bannerText").textContent = text;
    $("banner").hidden = false;
  }

  function showLockout() {
    lockedOut = true;
    showBanner("Another browser tab controls this game. Close it, or reload this page to take over.");
  }

  // -- the view of the board ----------------------------------------------------------------------------
  function shownCandidates(moves, mover) {
    const total = moves.reduce((t, m) => t + (m.visits > 0 ? m.visits : 0), 0) || 1;
    return moves.filter(m => m.visits > 0 && m.visits / total > MIN_VISIT_SHARE)
      .map(m => ({
        move: m.move, visits: m.visits, lcb: m.lcb, prior: m.prior, utility: m.utility,
        win: m.value ? m.value[mover] : null, value: m.value, pv: m.pv,
      }))
      .sort((a, b) => (b.lcb - a.lcb) || (b.visits - a.visits))
      .slice(0, SHOWN_CANDIDATES);
  }

  function buildView() {
    const pos = state.position;
    const events = state.events.slice(0, state.cursor);
    const last = events.length ? events[events.length - 1] : null;
    let lastMove = null;
    if (last && last.a) {
      const p = C.parseMove(last.a);
      if (p) lastMove = p;
    }
    const dotSeat = selectedSeat !== null ? selectedSeat : pos.toMove;
    let legal = [];
    if (dotSeat === pos.toMove) legal = pos.legal || [];
    else if (pendingLegal && pendingLegal.seat === dotSeat) legal = pendingLegal.legal;
    const legalPawns = new Set(legal.filter(m => C.parseMove(m) && C.parseMove(m).kind === "pawn"));
    // Walls can only be placed by the seat to move: the wall preview is for that seat.
    const legalWalls = new Set(dotSeat === pos.toMove ? (pos.legal || []).filter(m => m.endsWith("h") || m.endsWith("v")) : []);
    const an = state.analysis;
    let candidates = [];
    if (an.on && an.moves && an.moves.length && an.ply === state.cursor) candidates = shownCandidates(an.moves, pos.toMove);
    const colors = C.SEAT_COLORS;
    return {
      position: pos,
      colors,
      wallSeats: C.wallSeats(events.map(e => Object.assign({ applied: true }, e))),
      lastMove,
      dotSeat,
      legalDots: [...legalPawns],
      legalPawns,
      legalWalls,
      candidates,
      pv: pvOverlay,
      inactive: pos.finished,
    };
  }

  function renderBoard() {
    board.render(buildView());
  }

  // -- header and banner ---------------------------------------------------------------------------------
  function renderHeader() {
    const s = state;
    $("mode").textContent = s.mode === "analysis" ? "analysis (all human)" : "play";
    $("opp").textContent = "opponent: " + s.settings.opponentMode;
    const eng = s.engine;
    const el = $("engine");
    el.textContent = eng.status === "ready" ? "engine ready" : "engine down";
    el.className = "badge " + (eng.status === "ready" ? "ok" : "bad");
    if (eng.status !== "ready") {
      showBanner("The engine is not running: " + eng.message + ". Restart it to continue (the game is replayed).");
    } else if (!lockedOut) {
      $("banner").hidden = true;
    }
    const pos = s.position;
    if (pos) {
      const to = pos.finished
        ? (pos.winner >= 0 ? "game over: " + SEAT[pos.winner] + " wins" : "game over: draw")
        : "to move: " + SEAT[pos.toMove] + " (" + kindLabel(s.seats[pos.toMove]) + ")";
      $("toMoveText").textContent = to;
    }
  }

  function kindLabel(seat) {
    if (seat.kind === "human") return "human";
    if (seat.kind === "net") return "AI net, " + (seat.visits || state.settings.maxVisits) + " visits";
    return "bot " + seat.bot + (seat.target !== null && seat.target !== undefined ? " → " + SEAT[seat.target] : "");
  }

  // -- seat cards -------------------------------------------------------------------------------------------
  function buildSeatCards() {
    const box = $("seats");
    box.textContent = "";
    seatCards = [];
    for (let i = 0; i < 4; i++) {
      const card = document.createElement("div");
      card.className = "seat";
      card.style.borderLeftColor = C.SEAT_COLORS[i];
      card.innerHTML = '<div class="name"><span></span><span class="kind"></span></div>' +
        '<div class="stats"></div>' +
        '<select aria-label="kind"><option value="human">Human</option><option value="net">AI net</option><option value="bot">Bot</option></select>' +
        '<div class="sub-row"><input type="number" min="0" placeholder="visits (default)" aria-label="visits" hidden>' +
        '<select aria-label="bot" hidden></select><select aria-label="grudge target" hidden></select></div>' +
        '<div class="sub-row"><button type="button" class="danger elim">eliminate</button></div>';
      box.appendChild(card);
      const cell = { el: card, i };
      cell.name = card.querySelector(".name span");
      cell.kind = card.querySelector(".kind");
      cell.stats = card.querySelector(".stats");
      cell.kindSel = card.querySelector("select");
      cell.visits = card.querySelector('input[type="number"]');
      const subs = card.querySelectorAll(".sub-row select");
      cell.botSel = subs[0];
      cell.targetSel = subs[1];
      cell.elim = card.querySelector(".elim");
      cell.botSel.innerHTML = BOTS.map(b => '<option value="' + b + '">' + b + "</option>").join("");
      cell.targetSel.innerHTML = '<option value="">target…</option>' + SEAT.map((l, k) =>
        '<option value="' + k + '">' + l + "</option>").join("");
      cell.kindSel.addEventListener("change", () => sendSeats());
      cell.visits.addEventListener("change", () => sendSeats());
      cell.botSel.addEventListener("change", () => sendSeats());
      cell.targetSel.addEventListener("change", () => sendSeats());
      cell.elim.addEventListener("click", () => eliminate(i));
      seatCards.push(cell);
    }
  }

  function renderSeats() {
    const s = state, pos = s.position;
    const alive = pos.alive.filter(Boolean).length;
    seatCards.forEach((cell) => {
      const i = cell.i, seat = s.seats[i];
      const isAlive = pos.alive[i];
      cell.el.classList.toggle("tomove", !pos.finished && pos.toMove === i);
      cell.el.classList.toggle("dead", !isAlive);
      cell.name.textContent = SEAT[i] + " · " + C.SEAT_NAMES[i] + (isAlive ? "" : " (eliminated)");
      cell.kind.textContent = kindLabel(seat);
      cell.stats.textContent = "walls left " + pos.wallsLeft[i] + " · distance " +
        (isAlive && pos.dist[i] < 255 ? pos.dist[i] : "–") + (isAlive ? "" : " · out");
      setIfIdle(cell.kindSel, seat.kind);
      cell.visits.hidden = seat.kind !== "net";
      setIfIdle(cell.visits, seat.visits ? String(seat.visits) : "", "visits (default)");
      cell.botSel.hidden = seat.kind !== "bot";
      cell.targetSel.hidden = !(seat.kind === "bot" && seat.bot === "grudge");
      setIfIdle(cell.botSel, seat.bot);
      setIfIdle(cell.targetSel, seat.target === null || seat.target === undefined ? "" : String(seat.target));
      cell.elim.disabled = !isAlive || pos.finished || busy;
    });
    $("anaCs").value = state.analysis.cs;
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
    return act("setting seats…", () => api("POST", "/api/seats", body));
  }

  async function eliminate(seat) {
    const alive = state.position.alive.filter(Boolean).length;
    let text = "Eliminate " + SEAT[seat] + " (" + C.SEAT_NAMES[seat] + ")? Its walls stay on the board; its unplaced walls "
      + "are lost. Undo brings it back.";
    let confirmed = false;
    if (alive === 2) {
      text += "\n\nThis leaves ONE seat: the game ends at once (undo reopens it).";
      confirmed = true;
    }
    if (!window.confirm(text)) return;
    await act("eliminating…", () => api("POST", "/api/eliminate", { seat, confirm: confirmed }));
  }

  // -- root values and the raw net -----------------------------------------------------------------------
  function renderRoot() {
    const an = state.analysis, pos = state.position;
    let p = null, src = "";
    if (an.root && an.root.value && an.ply === state.cursor) { p = an.root.value; src = "search"; }
    else if (pos.eval && pos.eval.p) { p = pos.eval.p; src = "stored"; }
    else if (pos.raw && pos.raw.value) { p = pos.raw.value; src = "net"; }
    const bar = $("rootBar");
    bar.textContent = "";
    const legend = $("rootLegend");
    legend.textContent = "";
    if (p) {
      const seg = C.barSegments(p);
      const colors = C.SEAT_COLORS.concat([C.DRAW_COLOR]);
      seg.forEach((f, k) => {
        const d = document.createElement("div");
        d.style.width = (100 * f) + "%";
        d.style.background = colors[k];
        d.title = (k < 4 ? SEAT[k] : "draw") + " " + C.pct(p[k]);
        bar.appendChild(d);
        const it = document.createElement("span");
        const sw = document.createElement("i");
        sw.style.background = colors[k];
        it.appendChild(sw);
        it.appendChild(document.createTextNode((k < 4 ? SEAT[k] : "draw") + " " + C.pct(p[k])));
        legend.appendChild(it);
      });
      legend.appendChild(document.createTextNode("  (" + (src === "search" ? "search" : src === "stored" ? "stored eval" : "raw net") + ")"));
    }
    const r = an.root;
    $("rootInfo").textContent = r && an.ply === state.cursor
      ? "root visits " + r.visits + " · utility " + fmtList(r.utility, 3) + " · lead " + fmtList(r.lead, 2) + " cells"
      : (an.on ? "searching…" : "analysis off: press Start or Space to search this position");
    const raw = pos.raw;
    $("rawInfo").textContent = raw
      ? "net, no search: " + SEAT.map((l, k) => l + " " + C.pct(raw.value[k])).join(" ") +
        " draw " + C.pct(raw.value[4]) + (raw.plies ? " · plies to end ≈ " + Math.round(raw.plies) : "")
      : "";
  }

  function fmtList(xs, d) {
    if (!xs) return "–";
    return xs.map(x => x.toFixed(d)).join(" ");
  }

  // -- candidates table ----------------------------------------------------------------------------------
  function renderCands() {
    const an = state.analysis, pos = state.position;
    const body = $("cands").querySelector("tbody");
    body.textContent = "";
    if (!an.moves || !an.moves.length || an.ply !== state.cursor) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");
      td.colSpan = 7;
      td.textContent = an.on ? "no candidates yet" : "";
      tr.appendChild(td);
      body.appendChild(tr);
      return;
    }
    const rows = shownCandidates(an.moves, pos.toMove);
    const sorted = C.sortCandidates(rows.map(r => Object.assign({}, r)), sortKey === "win" ? "lcb" : sortKey);
    if (sortKey === "win") sorted.sort((a, b) => (b.win - a.win) || (b.visits - a.visits));
    const best = rows[0];   // the engine's choice by LCB among the shown moves
    sorted.forEach((r) => {
      const tr = document.createElement("tr");
      if (best && r.move === best.move) tr.className = "best";
      addCell(tr, r.move);
      addCell(tr, String(r.visits));
      addCell(tr, C.pct(r.win));
      addCell(tr, C.pct(r.prior));
      addCell(tr, r.utility.toFixed(3));
      addCell(tr, r.lcb.toFixed(3));
      const tdBar = document.createElement("td");
      if (r.value) {
        const stack = document.createElement("div");
        stack.className = "stack";
        const cols = C.SEAT_COLORS.concat([C.DRAW_COLOR]);
        C.barSegments(r.value).forEach((f, k) => {
          const d = document.createElement("div");
          d.style.width = (100 * f) + "%";
          d.style.background = cols[k];
          d.title = (k < 4 ? SEAT[k] : "draw") + " " + C.pct(r.value[k]);
          stack.appendChild(d);
        });
        tdBar.appendChild(stack);
        tdBar.appendChild(document.createTextNode(" " + r.value.map(x => C.pct(x, 0)).join(" ")));
      }
      tr.appendChild(tdBar);
      tr.addEventListener("click", () => playMove(r.move));
      tr.addEventListener("mouseenter", () => {
        pvOverlay = (r.pv || []).map(([letter, move]) => ({ seat: SEAT.indexOf(letter), move }));
        $("pvText").textContent = (r.pv || []).map((step, k) => (k + 1) + ". " + step[0] + " " + step[1]).join("   ");
        renderBoard();
      });
      tr.addEventListener("mouseleave", () => {
        pvOverlay = null;
        $("pvText").textContent = "";
        renderBoard();
      });
      body.appendChild(tr);
    });
  }

  function addCell(tr, text) {
    const td = document.createElement("td");
    td.textContent = text;
    tr.appendChild(td);
  }

  // -- move list -----------------------------------------------------------------------------------------
  function renderMoves() {
    const ol = $("moves");
    ol.textContent = "";
    const start = document.createElement("li");
    start.innerHTML = '<span class="n">0</span><span>start</span>';
    if (state.cursor === 0) start.className = "cur";
    start.addEventListener("click", () => goto(0));
    ol.appendChild(start);
    let ply = 0;
    state.events.forEach((e, k) => {
      const li = document.createElement("li");
      const n = document.createElement("span");
      n.className = "n";
      const txt = document.createElement("span");
      if (e.elim) {
        li.classList.add("elim");
        n.textContent = "";
        txt.textContent = "EL " + SEAT[e.seat];
      } else {
        ply += 1;
        n.textContent = ply + ".";
        txt.textContent = SEAT[e.seat] + " " + e.a;
        if (e.ai) li.classList.add("ai");
        const pe = state.evals[k];
        if (pe) {
          const ev = document.createElement("span");
          ev.className = "ev";
          ev.textContent = "  " + C.pct(pe[e.seat], 0);
          txt.appendChild(ev);
        }
      }
      if (!e.applied) li.classList.add("unapplied");
      if (k === state.cursor - 1) li.classList.add("cur");
      li.appendChild(n);
      li.appendChild(txt);
      li.addEventListener("click", () => goto(k + 1));
      ol.appendChild(li);
    });
    // Keep the current row visible inside the list only (scrolling the page on every poll would be unusable).
    if (state.cursor !== lastCursorShown) {
      lastCursorShown = state.cursor;
      const cur = ol.querySelector(".cur");
      if (cur) ol.scrollTop = Math.max(0, cur.offsetTop - ol.clientHeight / 2);
    }
  }

  // -- top-level render -----------------------------------------------------------------------------------
  function render() {
    if (!state) return;
    renderHeader();
    renderBoard();
    renderSeats();
    renderRoot();
    renderCands();
    renderMoves();
    renderAnalysis();
    renderNav();
    if (!settingsFilled) fillSettings();
  }

  function renderAnalysis() {
    const an = state.analysis;
    $("anaBtn").textContent = an.on ? "Pause analysis" : "Start analysis";
    $("anaBtn").classList.toggle("primary", !an.on);
    $("anaState").textContent = an.on ? (an.running ? "running" : an.done ? "done (visit cap)" : "paused") : "off";
    if (document.activeElement !== $("anaCap")) $("anaCap").value = an.cap ? String(an.cap) : "";
  }

  function renderNav() {
    const cur = state.cursor, len = state.length;
    $("plyText").textContent = "position " + cur + " of " + len + (cur < len ? " (review: play to branch)" : "");
    $("undoBtn").disabled = busy || cur === 0;
    $("redoBtn").disabled = busy || cur >= len;
    $("homeBtn").disabled = busy || cur === 0;
    $("endBtn").disabled = busy || cur >= len;
    $("aiNowBtn").disabled = busy || !isAiToMove();
    $("aiAllBtn").disabled = busy || !isAiToMove();
    $("moveBtn").disabled = busy;
  }

  function isAiToMove() {
    if (!state || !state.position || state.position.finished) return false;
    return state.seats[state.position.toMove].kind !== "human";
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
    $("anaCs").value = state.analysis.cs;
    settingsFilled = true;
  }

  // -- actions -----------------------------------------------------------------------------------------------
  function playMove(text) {
    pendingLegal = null;
    selectedSeat = null;
    return act("playing " + text + "…", () => api("POST", "/api/play", { move: text }));
  }

  function goto(ply) {
    if (!state) return;
    const target = Math.max(0, Math.min(state.length, ply));
    if (target === state.cursor) return;
    return act("reviewing…", () => api("POST", "/api/goto", { ply: target }));
  }

  function undo() { return act("undo…", () => api("POST", "/api/undo")); }
  function redo() { return act("redo…", () => api("POST", "/api/redo")); }

  async function aiNow() {
    return act("the AI is thinking…", () => api("POST", "/api/ai", { all: false }));
  }

  async function aiUntilHuman() {
    if (aiRunning) return;
    aiRunning = true;
    $("aiStopBtn").hidden = false;
    try {
      for (;;) {
        let r;
        try {
          r = await api("POST", "/api/ai", { all: true });
        } catch (e) {
          toast(e.message, true);
          if (e.engineDown) showBanner(e.message);
          break;
        }
        await refresh();
        if (!r.more || !aiRunning) break;
      }
    } finally {
      aiRunning = false;
      $("aiStopBtn").hidden = true;
      await refresh().catch(() => {});
    }
  }

  function toggleAnalysis() {
    if (!state) return;
    return act(state.analysis.on ? "pausing analysis…" : "starting analysis…",
      () => api("POST", "/api/analysis", { on: !state.analysis.on }));
  }

  function applyAnalysisLimits() {
    const cs = parseInt($("anaCs").value, 10);
    const capText = $("anaCap").value.trim();
    const cap = capText === "" ? 0 : parseInt(capText, 10);
    if (!(cs >= 1) || (capText !== "" && !(cap >= 1))) {
      toast("interval must be a number of centiseconds (≥ 1); max visits empty or ≥ 1", true);
      return;
    }
    return act("applying…", () => api("POST", "/api/analysis", { cs, cap }));
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
    const me = $("setMe").value === "" ? null : parseInt($("setMe").value, 10);
    const same = state.me === me || (state.me == null && me == null);
    return act("applying settings…", async () => {
      await api("POST", "/api/settings", body);
      if (!same) await api("POST", "/api/seats", { seats: collectSeats(), me });
    }).then((r) => { settingsFilled = false; return r; });
  }

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
    const r = C.normaliseMoveInput($("moveBox").value);
    if (r.error) {
      $("moveMsg").textContent = r.error;
      $("moveMsg").classList.add("err");
      return;
    }
    $("moveMsg").classList.remove("err");
    $("moveMsg").textContent = "";
    $("moveBox").value = "";
    playMove(r.move);
  }

  async function newGame() {
    if (state.length > 0 && !window.confirm("Start a new game? Download the current game first if you want to keep it.")) return;
    $("dlgNew").showModal();
  }

  function startNewGame() {
    const rules = {
      maxPlies: parseInt($("newPlies").value, 10),
      repetitionDrawCount: parseInt($("newRep").value, 10),
      initialWalls: [0, 1, 2, 3].map(k => parseInt($("newW" + k).value, 10)),
    };
    $("dlgNew").close();
    selectedSeat = null;
    pendingLegal = null;
    return act("new game…", () => api("POST", "/api/new", { rules, seats: collectSeats() }));
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
    } catch (e) {
      toast(e.message, true);
    }
  }

  function loadGame() {
    const text = $("loadText").value;
    const plyText = $("loadPly").value.trim();
    const body = { text };
    if (plyText !== "") body.ply = parseInt(plyText, 10);
    $("dlgLoad").close();
    return act("loading…", () => api("POST", "/api/load", body));
  }

  async function restart() {
    $("bannerText").textContent = "Restarting the engine…";
    await act("restarting the engine…", () => api("POST", "/api/restart"));
  }

  function rotateTo(r) {
    rot = ((r % 4) + 4) % 4;
    board.setRotation(rot);
    document.querySelectorAll("#seatView button").forEach((b, k) => b.classList.toggle("on", k === rot));
  }

  // A pawn of another seat was clicked: its legal moves (a read from the engine), shown as dots.
  async function clickCell(name) {
    const pos = state.position;
    const seat = [0, 1, 2, 3].find(s => pos.pawns[s] === name && pos.alive[s]);
    if (seat === undefined) {
      if (selectedSeat !== null) { selectedSeat = null; pendingLegal = null; renderBoard(); }
      return;
    }
    selectedSeat = seat;
    if (seat === pos.toMove) { pendingLegal = null; renderBoard(); return; }
    try {
      const r = await api("GET", "/api/legal?seat=" + seat);
      pendingLegal = { seat, legal: r.legal };
    } catch (e) {
      toast(e.message, true);
      pendingLegal = null;
    }
    renderBoard();
  }

  // -- wiring ---------------------------------------------------------------------------------------------------
  function wire() {
    board = new Q4Board($("board"), {
      onMove: (name) => playMove(name),
      onCell: (name) => clickCell(name),
      onIllegal: (name) => { $("moveMsg").textContent = name + " is not legal here"; $("moveMsg").classList.add("err"); },
      onHover: (h) => {
        if (!h) { $("moveMsg").textContent = ""; $("moveMsg").classList.remove("err"); return; }
        if (h.kind === "wall") {
          $("moveMsg").textContent = h.name + (h.legal ? "" : " (illegal)");
          $("moveMsg").classList.toggle("err", !h.legal);
        }
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
    $("rotateBtn").addEventListener("click", () => rotateTo(rot + 1));
    $("moveBtn").addEventListener("click", submitMove);
    $("moveBox").addEventListener("keydown", (e) => { if (e.key === "Enter") submitMove(); });
    $("undoBtn").addEventListener("click", undo);
    $("redoBtn").addEventListener("click", redo);
    $("homeBtn").addEventListener("click", () => goto(0));
    $("endBtn").addEventListener("click", () => goto(state.length));
    $("aiNowBtn").addEventListener("click", aiNow);
    $("aiAllBtn").addEventListener("click", aiUntilHuman);
    $("aiStopBtn").addEventListener("click", () => { aiRunning = false; });
    $("anaBtn").addEventListener("click", toggleAnalysis);
    $("anaCapBtn").addEventListener("click", applyAnalysisLimits);
    $("setApply").addEventListener("click", applySettings);
    $("paramsBtn").addEventListener("click", showParams);
    $("saveBtn").addEventListener("click", saveSgf);
    $("loadBtn").addEventListener("click", () => $("dlgLoad").showModal());
    $("newBtn").addEventListener("click", newGame);
    $("newGo").addEventListener("click", startNewGame);
    $("loadGo").addEventListener("click", loadGame);
    $("loadFile").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = () => { $("loadText").value = String(reader.result).split(/\r?\n/).find(l => l.trim()) || ""; };
      reader.readAsText(f);
    });
    $("restartBtn").addEventListener("click", restart);
    document.querySelectorAll("#cands thead th[data-key]").forEach((th) => {
      th.addEventListener("click", () => { sortKey = th.dataset.key; renderCands(); });
    });
    const moves = $("moves");
    moves.addEventListener("wheel", (e) => {
      if (!state) return;
      e.preventDefault();
      goto(state.cursor + (e.deltaY > 0 ? 1 : -1));
    }, { passive: false });
    document.addEventListener("keydown", (e) => {
      const tag = (e.target && e.target.tagName) || "";
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag) || document.querySelector("dialog[open]")) return;
      if (!state) return;
      if (e.key === "ArrowLeft" || e.key === "u" || e.key === "U" || e.key === "Backspace") { e.preventDefault(); undo(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); redo(); }
      else if (e.key === "Home") { e.preventDefault(); goto(0); }
      else if (e.key === "End") { e.preventDefault(); goto(state.length); }
      else if (e.key === " ") { e.preventDefault(); toggleAnalysis(); }
      else if (e.key === "r" || e.key === "R") rotateTo(rot + 1);
    });
    buildSeatCards();
  }

  async function poll() {
    try {
      await refresh();
    } catch (e) {
      if (e.status === 423) showLockout();
      else if (e.status === 400) showBanner("This page did not identify itself; reload it.");
      else showBanner(e.message);
    }
    setTimeout(poll, 700);
  }

  wire();
  poll();
})();
