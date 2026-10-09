"use strict";
// Pure logic of the Q4 viewer: SGF parsing, replay, shortest paths, per-game summaries, scores. No DOM here, so the
// file can be loaded by tests/check_core.js. The format is docs/q4/Q4Sgf.md.
const SEATS = "SWNE";                       // seat 0..3 = S (f1), W (a6), N (f11), E (k6)
const SEAT_NAMES = ["South", "West", "North", "East"];
const N = 11, CENTER = [5, 5];
const START = [[5, 0], [0, 5], [5, 10], [10, 5]];
const DIRS = [[0, 1], [1, 0], [0, -1], [-1, 0]];

// ---- text ------------------------------------------------------------------
// Splits one SGF line into nodes [[id, value], ...]; a value escapes ] and \ with a backslash.
function parseNodes(line) {
  const s = line.trim();
  if (s[0] !== "(" || s[s.length - 1] !== ")") throw new Error("not an SGF line");
  const nodes = [];
  let i = 1;
  const n = s.length - 1;
  while (i < n) {
    const c = s[i];
    if (c === ";") { nodes.push([]); i++; }
    else if (c === " " || c === "\t") i++;
    else if (c >= "A" && c <= "Z") {
      if (!nodes.length) throw new Error("property before the first node");
      let j = i;
      while (j < n && s[j] >= "A" && s[j] <= "Z") j++;
      const id = s.slice(i, j);
      if (s[j] !== "[") throw new Error("property " + id + " has no value");
      j++;
      let v = "";
      for (;;) {
        if (j >= n + 1) throw new Error("unterminated value of " + id);
        if (s[j] === "\\" && j + 1 <= n) { v += s[j + 1]; j += 2; }
        else if (s[j] === "]") { j++; break; }
        else v += s[j++];
      }
      nodes[nodes.length - 1].push([id, v]);
      i = j;
    } else throw new Error("unexpected character " + c);
  }
  if (!nodes.length) throw new Error("no root node");
  return nodes;
}
function parseKV(text) {
  const kv = {};
  for (const part of text.split(",")) {
    const e = part.indexOf("=");
    if (e > 0) kv[part.slice(0, e).trim()] = part.slice(e + 1).trim();
  }
  return kv;
}
// "0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00" -> {p:[5], visits, weight}, or null if it is not a move comment
function parseMoveComment(text) {
  const parts = (text || "").trim().split(/\s+/);
  if (parts.length < 6) return null;
  const p = parts.slice(0, 5).map(Number);
  if (p.some(x => !isFinite(x) || x < 0 || x > 1.0001)) return null;
  let visits = null, weight = 0;
  for (const part of parts.slice(5)) {
    const e = part.indexOf("=");
    if (e < 0) continue;
    if (part.slice(0, e) === "v") visits = +part.slice(e + 1);
    else if (part.slice(0, e) === "weight") weight = +part.slice(e + 1);
  }
  return visits === null || !isFinite(visits) ? null : { p, visits, weight };
}

// ---- header ----------------------------------------------------------------
// The root node of a line, parsed on its own (bracket-aware scan to the second ';'): the cheap pass over a big file.
function rootEnd(line) {
  let inV = false;
  for (let i = 1; i < line.length; i++) {
    const c = line[i];
    if (inV) { if (c === "\\") i++; else if (c === "]") inV = false; }
    else if (c === "[") inV = true;
    else if (c === ";" && i > 1) return i;
  }
  return line.length - 1;
}
function headerOf(line) {
  const end = rootEnd(line);
  const nodes = parseNodes(line.slice(0, end) + ")");
  const root = {};
  for (const [k, v] of nodes[0]) root[k] = v;
  if (root.GM !== "Q4") {
    throw new Error(root.GM === "1" ? "This is a Duel (GM[1]) file: open it with python/sgfs_viewer" : "Not a Q4 game (GM[" + (root.GM ?? "") + "])");
  }
  const kv = parseKV(root.C || "");
  const ru = parseKV((root.RU || "").replace(/^Q4:/, ""));
  const rest = line.slice(end);
  const plies = (rest.match(/;[SWNE]\[/g) || []).length;
  const elims = (rest.match(/;EL\[/g) || []).length;
  const ids = ["PS", "PW", "PN", "PE"], wids = ["WS", "WW", "WN", "WE"];
  const h = {
    names: ids.map((id, i) => root[id] ?? "P" + (i + 1)),
    walls: wids.map(id => (root[id] === undefined ? 7 : +root[id])),
    maxPlies: ru.maxPlies === undefined ? 400 : +ru.maxPlies,
    rep: ru.repetitionDrawCount === undefined ? 0 : +ru.repetitionDrawCount,
    re: root.RE ?? "?",
    dr: root.DR ?? "",
    gtype: kv.gtype ?? "normal",
    startTurnIdx: +(kv.startTurnIdx ?? 0),
    gameHash: kv.gameHash ?? "",
    types: [0, 1, 2, 3].map(i => kv["type" + i] ?? ""),
    nets: [0, 1, 2, 3].map(i => kv["net" + i] ?? ""),
    visits: [0, 1, 2, 3].map(i => +(kv["v" + i] ?? 0)),
    table: kv.table ?? "", opening: kv.opening === undefined ? -1 : +kv.opening, rotation: kv.rotation === undefined ? -1 : +kv.rotation,
    plies, elims,
  };
  h.winner = SEATS.indexOf(h.re) >= 0 && h.re.length === 1 ? SEATS.indexOf(h.re) : -1;
  h.finished = h.re !== "?";
  h.isDraw = h.re === "0";
  h.drawReason = h.isDraw ? (h.dr || "draw") : (h.finished ? "" : "unfinished");
  return h;
}
// result category for filters: S W N E, draw-rep, draw-max, draw (other), unfinished
function resultKey(h) {
  if (h.winner >= 0) return SEATS[h.winner];
  if (h.isDraw) return h.drawReason === "repetition" ? "draw-rep" : h.drawReason === "maxPlies" ? "draw-max" : "draw";
  return "unfinished";
}

// ---- game ------------------------------------------------------------------
function parseGame(line) {
  const h = headerOf(line);
  const nodes = parseNodes(line);
  const events = [];
  for (const node of nodes.slice(1)) {
    let ev = null, c = "";
    for (const [k, v] of node) {
      if (k === "EL") ev = { elim: true, seat: SEATS.indexOf(v), text: "EL" };
      else if (k.length === 1 && SEATS.indexOf(k) >= 0) ev = { elim: false, seat: SEATS.indexOf(k), text: v };
      else if (k === "C") c = v;
    }
    if (!ev || ev.seat < 0) throw new Error("node without a move");
    ev.comment = parseMoveComment(c);
    ev.rawComment = c;
    events.push(ev);
  }
  const g = { h, events };
  buildStates(g);
  return g;
}
const cellOf = s => ({ x: s.charCodeAt(0) - 97, y: +s.slice(1) - 1 });
function parseAction(text) {
  const last = text[text.length - 1];
  if (last === "h" || last === "v") return { wall: true, h: last === "h", x: text.charCodeAt(0) - 97, y: +text.slice(1, -1) - 1 };
  const c = cellOf(text);
  return { wall: false, x: c.x, y: c.y };
}
// states[k] = the position after k events: pawns, walls left, alive, number of walls on the board. g.walls = every
// wall in placement order {h, x, y, seat}, g.states[k].nw of them are on the board after k events.
function buildStates(g) {
  const walls = [];
  const cur = { pawns: START.map(p => p.slice()), left: g.h.walls.slice(), alive: [true, true, true, true], nw: 0, last: null, toMove: 0 };
  const snap = last => ({ pawns: cur.pawns.map(p => p && p.slice()), left: cur.left.slice(), alive: cur.alive.slice(), nw: walls.length, last, toMove: cur.toMove });
  g.states = [snap(null)];
  let plies = 0;
  g.events.forEach((ev, i) => {
    ev.ply = ev.elim ? null : ++plies;
    if (ev.elim) { cur.alive[ev.seat] = false; cur.pawns[ev.seat] = null; }
    else {
      const a = parseAction(ev.text);
      ev.action = a;
      if (a.wall) { walls.push({ h: a.h, x: a.x, y: a.y, seat: ev.seat }); cur.left[ev.seat]--; }
      else cur.pawns[ev.seat] = [a.x, a.y];
    }
    // the next seat to move: the next alive seat after the one that moved (a skipped seat is not recorded)
    let nxt = ev.elim ? cur.toMove : (ev.seat + 1) % 4;
    for (let t = 0; t < 4 && !cur.alive[nxt]; t++) nxt = (nxt + 1) % 4;
    if (ev.elim && cur.toMove === ev.seat) { nxt = (ev.seat + 1) % 4; for (let t = 0; t < 4 && !cur.alive[nxt]; t++) nxt = (nxt + 1) % 4; }
    cur.toMove = nxt;
    g.states.push(snap({ index: i, seat: ev.seat, elim: ev.elim, action: ev.action }));
  });
  g.walls = walls;
}
// the seat to move at position k: that of the event k (the next one), else none
function moverAt(g, k) { return k < g.events.length ? g.events[k].seat : -1; }

// ---- shortest paths ----------------------------------------------------------
// Passage between adjacent cells (x0,y0) and (x1,y1) given the wall set keys "h:x:y" / "v:x:y".
function blocked(ws, x0, y0, x1, y1) {
  if (x0 === x1) { const y = Math.min(y0, y1); return ws.has("h:" + x0 + ":" + y) || ws.has("h:" + (x0 - 1) + ":" + y); }
  const x = Math.min(x0, x1);
  return ws.has("v:" + x + ":" + y0) || ws.has("v:" + x + ":" + (y0 - 1));
}
function wallSet(g, k) {
  const ws = new Set();
  for (let i = 0; i < g.states[k].nw; i++) { const w = g.walls[i]; ws.add((w.h ? "h:" : "v:") + w.x + ":" + w.y); }
  return ws;
}
// BFS distance over the cell graph (walls and edges only, pawns ignored) from (sx, sy); Int16Array[121], -1 unreachable.
function bfs(ws, sx, sy) {
  const dist = new Int16Array(N * N).fill(-1);
  const q = [sx + sy * N];
  dist[q[0]] = 0;
  for (let h = 0; h < q.length; h++) {
    const c = q[h], x = c % N, y = (c - x) / N;
    for (const [dx, dy] of DIRS) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= N || ny >= N || dist[nx + ny * N] >= 0 || blocked(ws, x, y, nx, ny)) continue;
      dist[nx + ny * N] = dist[c] + 1;
      q.push(nx + ny * N);
    }
  }
  return dist;
}
// {dist: [per seat distance or null], cells: [per seat Set of cell indexes on any shortest path]} at position k
function pathsAt(g, k) {
  const st = g.states[k], ws = wallSet(g, k);
  const fromCenter = bfs(ws, CENTER[0], CENTER[1]);
  const dist = [null, null, null, null], cells = [null, null, null, null];
  for (let s = 0; s < 4; s++) {
    if (!st.alive[s]) continue;
    const [px, py] = st.pawns[s], d = fromCenter[px + py * N];
    dist[s] = d;
    if (d < 0) continue;
    const fromPawn = bfs(ws, px, py), set = new Set();
    for (let c = 0; c < N * N; c++) if (fromPawn[c] >= 0 && fromCenter[c] >= 0 && fromPawn[c] + fromCenter[c] === d) set.add(c);
    cells[s] = set;
  }
  return { dist, cells };
}

// ---- scores ------------------------------------------------------------------
// points of seat s in a finished game: 1 for the win, 1/4 for a draw, 0 for a loss; null for an unfinished game
function seatPoints(h, s) { return !h.finished ? null : h.isDraw ? 0.25 : h.winner === s ? 1 : 0; }
function wilson(k, n, z = 1.96) {
  if (!n) return [0, 1];
  const p = k / n, d = 1 + z * z / n, c = p + z * z / (2 * n), m = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
  return [Math.max(0, (c - m) / d), Math.min(1, (c + m) / d)];
}

// ---- per-game summary (the background pass) -------------------------------------
// Everything the filters, sorts and statistics need, without keeping the parsed game.
function summarize(g) {
  const h = g.h, ev = g.events;
  const S = { plies: h.plies, elims: h.elims, comments: 0, visits: [], weights: [], winnerProb: [], swing: 0, early: null,
    calib: [], blunder: null, minWinnerProb: null, hasEval: false };
  let lo = [1, 1, 1, 1, 1], hi = [0, 0, 0, 0, 0];
  const lastBySeat = [null, null, null, null];
  let plyNo = 0, bestDrop = 0;
  ev.forEach((e, i) => {
    if (!e.elim) plyNo++;
    const c = e.comment;
    if (!c) return;
    S.comments++;
    S.hasEval = true;
    S.visits.push(c.visits); S.weights.push(c.weight);
    for (let t = 0; t < 5; t++) { lo[t] = Math.min(lo[t], c.p[t]); hi[t] = Math.max(hi[t], c.p[t]); }
    if (S.early === null && Math.max(c.p[0], c.p[1], c.p[2], c.p[3]) > 0.9) S.early = plyNo;
    if (h.winner >= 0) {
      S.winnerProb.push([plyNo, c.p[h.winner]]);
      S.minWinnerProb = S.minWinnerProb === null ? c.p[h.winner] : Math.min(S.minWinnerProb, c.p[h.winner]);
    }
    if (h.finished && !e.elim && (plyNo - 1) % 10 === 0) S.calib.push([c.p[e.seat], seatPoints(h, e.seat)]);
    const prev = lastBySeat[e.seat];
    if (prev) { const drop = prev.p - c.p[e.seat]; if (drop > bestDrop) { bestDrop = drop; S.blunder = prev.i; } }
    lastBySeat[e.seat] = { i, p: c.p[e.seat] };
  });
  for (let t = 0; t < 4; t++) if (hi[t] >= lo[t]) S.swing = Math.max(S.swing, hi[t] - lo[t]);
  S.blunderDrop = bestDrop;
  return S;
}
// the event index of a game's biggest own-win-probability drop (the viewer's "blunder"), from its events
function blunderOf(g) { return summarize(g).blunder; }

if (typeof module !== "undefined") module.exports = { parseNodes, parseKV, parseMoveComment, headerOf, parseGame, resultKey, pathsAt, bfs, wilson };
