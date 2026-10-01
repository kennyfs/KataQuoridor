"use strict";
// ---- Board geometry -------------------------------------------------------
// The 17x17 search board: pawn cells at (even, even), horizontal wall centers at (odd, odd),
// vertical walls encoded by their upper arm at (odd, even) with the center one row below.
// SGF coords: x = letter - 'a', y = letter - 'a'. Notation matches Location::toString:
// pawn "e8", walls "e2h"/"e2v" with col = x/2, row = y/2 + 1.
// Move comments (sgf.cpp) are the value targets of the search at the position BEFORE that
// move, from White's perspective: "win loss noResult score v=.. [rv=..] weight=.. [result=..] [lead=..]".
// score is the search's utility score u (with the time bonus: not comparable across nets trained with different
// lambdas) and is never displayed when lead is present; lead (scoreLead) is the predicted tempo lead s. ev.wMargin
// is the margin shown and used as a prediction: lead when present, else score (old files: ev.hasLead is false and the
// viewer labels it "margin (no lead in file)").
// So the eval of the position after `ply` moves is moves[ply].ev.
const N = 9, WALLS_PER_PLAYER = 10;
const CELL = 50, GAP = 12, PITCH = CELL + GAP, PAD = 30;
const BOARD_PX = N * CELL + (N - 1) * GAP;
const SVGNS = "http://www.w3.org/2000/svg";

let lines = [];       // raw lines (one game per line)
let games = [];       // light headers for the list
let filtered = [];    // indices into games
let cur = null;       // parsed current game
let curGameIdx = -1;
let ply = 0;          // number of moves applied
let flipped = false;
let showPaths = true;  // overlay all shortest paths on the board
let playTimer = null;
let statsCache = null; // per-game summaries (sort metrics + statistics), filled by computeSummaries()
// Match files (gatekeeper): every game is model X vs model Y with PB != PW. Self-play files have PB == PW.
// models[0] is the newer one (larger "-s<samples>" in the name) when both names carry it.
let match = null;      // { models: [newer, older], labels: ["New", "Old"] } or null

const $ = id => document.getElementById(id);

// ---- Parsing ---------------------------------------------------------------
function parseKV(s) {
  const kv = {};
  for (const part of (s || "").split(",")) { const [k, v] = part.split("="); if (k) kv[k.trim()] = (v ?? "").trim(); }
  return kv;
}

function headerOf(line, i) {
  const re = /RE\[([^\]]*)\]/.exec(line);
  const c = /C\[([^\]]*)\]/.exec(line);
  const kv = parseKV(c && c[1]);
  const nMoves = (line.match(/;[BW]\[/g) || []).length;
  const result = re ? re[1] : "?";
  const pb = (/PB\[([^\]]*)\]/.exec(line) || [])[1] || "", pw = (/PW\[([^\]]*)\]/.exec(line) || [])[1] || "";
  return { i, result, winner: result[0], nMoves, gtype: kv.gtype || "", startTurnIdx: +kv.startTurnIdx || 0, pb, pw,
    winnerModel: result[0] === "B" ? pb : result[0] === "W" ? pw : "", ...rulesOf(line) };
}

// Komi and initial walls (docs/QuoridorIOv2.md): KM (missing, or KM[0] as 0.1.0 wrote it: the standard -0.5),
// WB / WW (missing: 10). rulesLabel is "" for the standard game.
const STANDARD_KOMI = -0.5, STANDARD_WALLS = 10;
function rulesOf(line) {
  const semi = line.indexOf(";", 2), root = semi < 0 ? line : line.slice(0, semi);
  const prop = p => (new RegExp("(?<![A-Z])" + p + "\\[([^\\]]*)\\]").exec(root) || [])[1];
  let komi = parseFloat(prop("KM"));
  if (!isFinite(komi) || komi === 0) komi = STANDARD_KOMI;
  const wb = parseInt(prop("WB") ?? STANDARD_WALLS, 10), ww = parseInt(prop("WW") ?? STANDARD_WALLS, 10);
  const parts = [];
  if (komi !== STANDARD_KOMI) parts.push(`komi ${fmtKomi(komi)}`);
  if (wb !== STANDARD_WALLS || ww !== STANDARD_WALLS) parts.push(`walls ${wb}/${ww}`);
  return { komi, wb, ww, rulesLabel: parts.join(" · ") };
}
const fmtKomi = k => (k > 0 ? "+" : k < 0 ? "−" : "") + Math.abs(k);

function parseGame(line) {
  const semi = line.indexOf(";", 2);
  const rootStr = semi < 0 ? line : line.slice(0, semi);
  const props = {};
  for (const m of rootStr.matchAll(/([A-Z]+)((?:\[[^\]]*\])+)/g))
    props[m[1]] = [...m[2].matchAll(/\[([^\]]*)\]/g)].map(x => x[1]);
  const rootKV = parseKV(props.C?.[0]);

  const moves = [];
  const body = semi < 0 ? "" : line.slice(semi);
  for (const m of body.matchAll(/;([BW])\[([a-z]*)\](?:C\[([^\]]*)\])?/g)) {
    const pla = m[1], s = m[2];
    const mv = { pla, raw: s, ev: null };
    if (s.length === 2) { mv.x = s.charCodeAt(0) - 97; mv.y = s.charCodeAt(1) - 97; }
    else mv.pass = true;
    if (m[3] !== undefined) mv.ev = parseEval(m[3]);
    moves.push(mv);
  }
  const pt = str => ({ x: str.charCodeAt(0) - 97, y: str.charCodeAt(1) - 97 });
  return {
    props, kv: rootKV, moves,
    startB: props.AB ? pt(props.AB[0]) : { x: 8, y: 16 },
    startW: props.AW ? pt(props.AW[0]) : { x: 8, y: 0 },
    startTurnIdx: +rootKV.startTurnIdx || 0,
  };
}

function parseEval(c) {
  const t = c.trim().split(/\s+/);
  if (t.length < 4 || isNaN(+t[0])) {       // e.g. only "result=..." on an init move
    const ev = {}; for (const tok of t) { const [k, v] = tok.split("="); ev[k] = v; }
    return ev.result ? { onlyResult: true, result: ev.result } : null;
  }
  const ev = { wWin: +t[0], wLoss: +t[1], noRes: +t[2], wScore: +t[3] };
  for (const tok of t.slice(4)) { const [k, v] = tok.split("="); ev[k] = v; }
  if (ev.v !== undefined) ev.v = +ev.v;
  if (ev.rv !== undefined) ev.rv = +ev.rv;
  if (ev.weight !== undefined) ev.weight = +ev.weight;
  if (ev.lead !== undefined) ev.lead = +ev.lead;
  ev.hasLead = ev.lead !== undefined;
  ev.wMargin = ev.hasLead ? ev.lead : ev.wScore;
  return ev;
}
const hasEv = mv => mv && mv.ev && !mv.ev.onlyResult;

function moveKind(mv) {
  if (mv.pass) return "pass";
  const ox = mv.x & 1, oy = mv.y & 1;
  if (!ox && !oy) return "pawn";
  if (ox && oy) return "hwall";
  if (ox && !oy) return "vwall";
  return "?";
}

function notation(mv) {
  const k = moveKind(mv);
  if (k === "pass") return "pass";
  const s = String.fromCharCode(97 + (mv.x >> 1)) + ((mv.y >> 1) + 1);
  return k === "pawn" ? s : k === "hwall" ? s + "h" : k === "vwall" ? s + "v" : mv.raw;
}

// ---- State reconstruction --------------------------------------------------
function stateAt(g, n) {
  const st = { B: { ...g.startB }, W: { ...g.startW }, walls: [], used: { B: 0, W: 0 }, prevPawn: null, last: null };
  for (let i = 0; i < n; i++) {
    const mv = g.moves[i], k = moveKind(mv);
    if (k === "pawn") { if (i === n - 1) st.prevPawn = { ...st[mv.pla] }; st[mv.pla] = { x: mv.x, y: mv.y }; }
    else if (k === "hwall" || k === "vwall") { st.walls.push({ kind: k, x: mv.x, y: mv.y, pla: mv.pla, idx: i }); st.used[mv.pla]++; }
  }
  if (n > 0) st.last = g.moves[n - 1];
  st.toMove = n < g.moves.length ? g.moves[n].pla : (n > 0 ? (g.moves[n - 1].pla === "B" ? "W" : "B") : "B");
  st.block = newBlock();
  for (const w of st.walls) addWallToBlock(st.block, w.kind, w.x, w.y);
  return st;
}


// ---- Shortest paths (no jumps, pawns ignored) ------------------------------
// Pawn cell (c, r), c,r in 0..8, index r*9+c. Blocked edges are stored per cell:
// right[k] = edge (c,r)-(c+1,r) is walled, down[k] = edge (c,r)-(c,r+1) is walled.
// A vertical wall with upper arm (2c+1, 2r) blocks right-edges of (c,r) and (c,r+1);
// a horizontal wall centered at (2c+1, 2r+1) blocks down-edges of (c,r) and (c+1,r).
// Black's goal is row 0 (y=0), White's goal is row 8 (y=16).
const GOAL = { B: 0, W: N - 1 };

function newBlock() { return { right: new Uint8Array(N * N), down: new Uint8Array(N * N) }; }
function addWallToBlock(bl, kind, x, y) {
  const c = x >> 1, r = y >> 1;
  if (kind === "vwall") { bl.right[r * N + c] = 1; if (r + 1 < N) bl.right[(r + 1) * N + c] = 1; }
  else if (kind === "hwall") { bl.down[r * N + c] = 1; if (c + 1 < N) bl.down[r * N + c + 1] = 1; }
}
function neighbors(bl, k) {
  const c = k % N, r = (k / N) | 0, out = [];
  if (c + 1 < N && !bl.right[k]) out.push(k + 1);
  if (c > 0 && !bl.right[k - 1]) out.push(k - 1);
  if (r + 1 < N && !bl.down[k]) out.push(k + N);
  if (r > 0 && !bl.down[k - N]) out.push(k - N);
  return out;
}
const _bfsQ = new Int16Array(N * N);
function bfs(bl, sources) {
  const d = new Int16Array(N * N).fill(-1), q = _bfsQ, R = bl.right, D = bl.down;
  let tail = 0;
  for (const s of sources) { d[s] = 0; q[tail++] = s; }
  for (let h = 0; h < tail; h++) {
    const k = q[h], c = k % N, nd = d[k] + 1;
    if (c + 1 < N && !R[k] && d[k + 1] < 0) { d[k + 1] = nd; q[tail++] = k + 1; }
    if (c > 0 && !R[k - 1] && d[k - 1] < 0) { d[k - 1] = nd; q[tail++] = k - 1; }
    if (k + N < N * N && !D[k] && d[k + N] < 0) { d[k + N] = nd; q[tail++] = k + N; }
    if (k >= N && !D[k - N] && d[k - N] < 0) { d[k - N] = nd; q[tail++] = k - N; }
  }
  return d;
}
const GOAL_CELLS = { B: Array.from({ length: N }, (_, c) => GOAL.B * N + c), W: Array.from({ length: N }, (_, c) => GOAL.W * N + c) };
const goalDist = (bl, pla) => bfs(bl, GOAL_CELLS[pla]);
// Edges a wall blocks, as cell-index pairs. Distances to a goal can only change if one of them
// joins two cells whose goal distances differ (i.e. lies on some shortest path to that goal).
function wallEdges(kind, x, y) {
  const c = x >> 1, r = y >> 1, k = r * N + c;
  return kind === "vwall" ? [[k, k + 1], [k + N, k + N + 1]] : [[k, k + N], [k + 1, k + N + 1]];
}
const wallAffects = (d, kind, x, y) => wallEdges(kind, x, y).some(([u, v]) => d[u] !== d[v]);
const cellOf = pos => (pos.y >> 1) * N + (pos.x >> 1);

// All shortest paths from pos to pla's goal row: cells and directed edges of the shortest-path DAG.
function shortestPaths(bl, pos, pla) {
  const dG = goalDist(bl, pla), s = cellOf(pos), D = dG[s];
  if (D < 0) return { len: -1, cells: new Set(), edges: [] };
  const dS = bfs(bl, [s]), cells = new Set(), edges = [];
  for (let k = 0; k < N * N; k++) {
    if (dS[k] < 0 || dG[k] < 0 || dS[k] + dG[k] !== D) continue;
    cells.add(k);
    if (dG[k] === 0) continue;   // stop at the goal row
    for (const v of neighbors(bl, k)) if (dS[v] === dS[k] + 1 && dG[v] === dG[k] - 1) edges.push([k, v]);
  }
  return { len: D, cells, edges };
}
