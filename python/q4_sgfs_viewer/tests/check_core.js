// Checks the pure functions of js/core.js (header parsing, escapes, results and draws, comment parser, replay with an
// elimination, BFS distances, summaries) on hand-written lines and on real games:
//     node python/q4_sgfs_viewer/tests/check_core.js [file.sgfs]
// The default file is fixture.sgfs next to this script: ~12 games written by the real writers (self-play with a random
// net: normal / mixed / fork games, eliminations, a draw by maxPlies, a draw by repetition; q4match games with a
// search net and bots; a q4gatekeeper game).
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const src = fs.readFileSync(path.join(__dirname, "..", "js", "core.js"), "utf8");
const ctx = vm.createContext({ console });
vm.runInContext(src + "\nthis.api = { parseNodes, parseKV, parseMoveComment, headerOf, parseGame, resultKey, pathsAt, bfs, wilson, summarize, seatPoints, wallSet, SEATS };", ctx);
const A = ctx.api;
const plain = x => JSON.parse(JSON.stringify(x));   // values made inside the vm context are not reference-equal

// ---- parsing and escapes
const nodes = A.parseNodes("(;FF[4]GM[Q4]PS[a\\]b\\\\c];S[f2]C[0.1 0.2 0.3 0.4 0.0 v=5 weight=1.00];EL[W])");
assert.strictEqual(nodes.length, 3);
assert.strictEqual(nodes[0][2][1], "a]b\\c", "escaped ] and backslash");
assert.strictEqual(nodes[2][0][0], "EL");
assert.throws(() => A.parseNodes("(;S[f2"), /not an SGF/);
assert.throws(() => A.parseNodes("(;S[f2)"), /unterminated/);
assert.throws(() => A.parseNodes("nothing"), /not an SGF/);
assert.deepStrictEqual(JSON.parse(JSON.stringify(A.parseKV("a=1, b = x=y ,c"))), { a: "1", b: "x=y" });
const c = A.parseMoveComment("0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00");
assert.deepStrictEqual(JSON.parse(JSON.stringify(c)), { p: [0.31, 0.22, 0.27, 0.15, 0.05], visits: 600, weight: 1 });
assert.strictEqual(A.parseMoveComment("elim"), null);
assert.strictEqual(A.parseMoveComment("0.1 0.2 0.3 0.4 0.5 weight=1"), null, "no visits");
assert.strictEqual(A.parseMoveComment("0.5 0.2 0.3 0.4 0.5 v=1")?.p[0], 0.5);
assert.strictEqual(A.parseMoveComment("7 0.2 0.3 0.4 0.5 v=1"), null, "not a probability");

// ---- header: results, draws, rules, match metadata; a name with a bracket
const root = n => `(;FF[4]GM[Q4]SZ[11]PS[a\\]x]PW[b]PN[c]PE[d]WS[7]WW[7]WN[5]WE[7]RU[Q4:repetitionDrawCount=3,maxPlies=50]${n}`;
let h = A.headerOf(root("RE[N]C[gtype=fork,startTurnIdx=2,type0=selfplay,net0=/m/net1/model.bin.gz,v0=600,table=t,opening=3,rotation=1,gameHash=00ff];S[f2];W[b6])"));
assert.strictEqual(h.winner, 2); assert.strictEqual(A.resultKey(h), "N");
assert.deepStrictEqual(plain(h.walls), [7, 7, 5, 7]); assert.strictEqual(h.names[0], "a]x");
assert.strictEqual(h.rep, 3); assert.strictEqual(h.maxPlies, 50); assert.strictEqual(h.plies, 2);
assert.strictEqual(h.gtype, "fork"); assert.strictEqual(h.startTurnIdx, 2); assert.strictEqual(h.visits[0], 600);
assert.strictEqual(h.table, "t"); assert.strictEqual(h.opening, 3); assert.strictEqual(h.rotation, 1);
h = A.headerOf(root("RE[0]DR[repetition]C[gtype=normal];S[f2])"));
assert.ok(h.isDraw && h.winner === -1 && A.resultKey(h) === "draw-rep");
assert.strictEqual(A.resultKey(A.headerOf(root("RE[0]DR[maxPlies];S[f2])"))), "draw-max");
h = A.headerOf(root("RE[?]DR[unfinished];S[f2])"));
assert.ok(!h.finished && A.resultKey(h) === "unfinished" && A.seatPoints(h, 0) === null);
assert.strictEqual(A.seatPoints(A.headerOf(root("RE[0];S[f2])")), 0) , 0.25);
assert.throws(() => A.headerOf("(;FF[4]GM[1]SZ[17]PB[a])"), /Duel/);
assert.throws(() => A.headerOf("(;FF[4]SZ[11])"), /Not a Q4/);

// ---- replay with an elimination: the pawn is removed, the walls stay, the walls left count down
const g = A.parseGame(root("RE[S]C[gtype=normal];S[f2];W[b6];N[e10h]C[0.2 0.2 0.2 0.2 0.2 v=9 weight=0.50];EL[W];E[j6];S[f3])"));
assert.strictEqual(g.events.length, 6);
assert.strictEqual(g.events[2].comment.visits, 9);
let st = g.states[3];
assert.deepStrictEqual(plain(st.pawns[1]), [1, 5]); assert.strictEqual(st.left[2], 4); assert.strictEqual(st.nw, 1);
st = g.states[4];
assert.strictEqual(st.pawns[1], null); assert.strictEqual(st.alive[1], false); assert.strictEqual(st.nw, 1, "walls stay");
assert.strictEqual(g.states[3].toMove, 3, "E moves after N");
assert.strictEqual(g.states[0].toMove, 0);
assert.strictEqual(g.events[3].ply, null, "an elimination has no ply"); assert.strictEqual(g.events[4].ply, 4);
assert.throws(() => A.parseGame(root("RE[S];X[f2])")), /node without a move/);

// ---- BFS distances: empty board = Manhattan distance; walls lengthen paths; shortest-path cells
let p0 = A.pathsAt(A.parseGame(root("RE[?];S[f2])")), 0);
assert.deepStrictEqual(Array.from(p0.dist), [5, 5, 5, 5]);
assert.strictEqual(p0.cells[0].size, 6, "a straight line of 6 cells (f1..f6)");
p0 = A.pathsAt(A.parseGame(root("RE[?];S[e1])")), 0);
const gw = A.parseGame(root("RE[?];S[f2];W[b6];N[f9];E[j6];S[e2h])"));   // a wall above f2/e2 ... south pawn at f2
const after = A.pathsAt(gw, 5);
assert.strictEqual(after.dist[0], 6, "S at f2 had 4 steps; e2h blocks f2->f3 and e2->e3, the detour via g2 takes 6");
const d0 = A.pathsAt(gw, 1).dist[0];
assert.strictEqual(d0, 4);
// a corner cell: the distance of a1 is 10 on an empty board
const empty = A.bfs(new Set(), 5, 5);
assert.strictEqual(empty[0], 10); assert.strictEqual(empty[5 + 5 * 11], 0);
const walled = A.bfs(A.wallSet(gw, 5), 5, 5);
assert.strictEqual(walled[5 + 1 * 11], 6, "f2: the detour around the wall costs 2 more steps");

// ---- Wilson interval
const [lo, hi] = A.wilson(50, 100);
assert.ok(lo > 0.4 && lo < 0.41 && hi > 0.59 && hi < 0.6);
assert.deepStrictEqual(Array.from(A.wilson(0, 0)), [0, 1]);

// ---- the fixture: real games
const file = process.argv[2] || path.join(__dirname, "fixture.sgfs");
const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(l => l.startsWith("(;"));
assert(lines.length > 0, "no games");
let kinds = new Set();
for (const line of lines) {
  const hd = A.headerOf(line), game = A.parseGame(line);
  assert.strictEqual(game.events.filter(e => !e.elim).length, hd.plies, "header ply count = parsed plies");
  assert.strictEqual(game.events.filter(e => e.elim).length, hd.elims);
  kinds.add(A.resultKey(hd)); kinds.add("gtype:" + hd.gtype);
  if (hd.elims) kinds.add("elim");
  if (hd.table) kinds.add("match");
  // every position has a finite or -1 distance for the alive seats, and a dead seat has no pawn
  const last = game.states[game.events.length];
  const pa = A.pathsAt(game, game.events.length);
  for (let s = 0; s < 4; s++) assert.strictEqual(last.alive[s], pa.dist[s] !== null);
  if (hd.winner >= 0) assert.deepStrictEqual(plain(last.pawns[hd.winner]), [5, 5], "the winner stands on the centre");
  const sm = A.summarize(game);
  assert.strictEqual(sm.plies, hd.plies);
  assert.ok(sm.swing >= 0 && sm.swing <= 1);
  for (const e of game.events) if (e.comment) assert.ok(Math.abs(e.comment.p.reduce((a, b) => a + b, 0) - 1) < 0.03, "probabilities sum to 1");
}
console.log(`${lines.length} games ok; seen: ${[...kinds].sort().join(", ")}`);
for (const need of ["draw-rep", "draw-max", "elim", "match"]) assert.ok(kinds.has(need), "fixture lacks " + need);
console.log("check_core: all checks passed");
