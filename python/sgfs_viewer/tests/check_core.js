// Checks the viewer's pure functions (header parsing, results and draws, scores, the repetition-stretch metric,
// per-game summaries) on real SGF lines: node python/sgfs_viewer/tests/check_core.js [file.sgfs]
// Default file: fixture.sgfs next to this script (self-play rule on N = 3 / 4 and off, draws by repetition and by
// maxPlies, a fork, fence handicaps on each side, komi != -0.5, a gatekeeper and a kq_ladder game).
"use strict";
const fs = require("fs"), path = require("path"), vm = require("vm"), assert = require("assert");
const js = path.join(__dirname, "..", "js");
const ctx = vm.createContext({ console, document: {} });
vm.runInContext(["core.js", "stats.js"].map(f => fs.readFileSync(path.join(js, f), "utf8")).join("\n") +
  "\nthis.api = { headerOf, parseGame, scoreB, isDraw, repLabel, repeatStretch, summarize, bdw, openingOf, openingLabel, moveKind, OPENING_PLIES };", ctx);
const A = ctx.api;

const file = process.argv[2] || path.join(__dirname, "fixture.sgfs");
const lines = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(l => l.startsWith("(;"));
assert(lines.length > 0, "no games");

// Independent position count: how often each position (pawns, walls placed, side to move) occurs, max over the game,
// and the count of the final position.
function occurrences(g) {
  const pos = { B: `${g.startB.x},${g.startB.y}`, W: `${g.startW.x},${g.startW.y}` };
  let walls = 0;
  const cnt = new Map(), key = toMove => `${pos.B}|${pos.W}|${walls}|${toMove}`;
  const bump = k => { const c = (cnt.get(k) || 0) + 1; cnt.set(k, c); return c; };
  let max = bump(key(g.moves.length ? g.moves[0].pla : "B")), last = 1;
  for (const mv of g.moves) {
    const k = A.moveKind(mv);
    if (k === "pawn") pos[mv.pla] = `${mv.x},${mv.y}`; else if (k.endsWith("wall")) walls++;
    last = bump(key(mv.pla === "B" ? "W" : "B"));
    max = Math.max(max, last);
  }
  return { max, last };
}

const seen = { ruleOn: 0, ruleOff: 0, n3: 0, n4: 0, repDraw: 0, maxPliesDraw: 0, fork: 0, hcB: 0, hcW: 0, komi: 0, match: 0, decisive: 0 };
lines.forEach((line, i) => {
  const h = A.headerOf(line, i), g = A.parseGame(line), s = A.summarize(line, h);
  const re = /RE\[([^\]]*)\]/.exec(line)[1];
  assert.strictEqual(h.result, re);
  assert.strictEqual(h.nMoves, g.moves.length);
  // results and scores: a draw (RE[0]) is half a point, never a White win
  if (re === "0") {
    assert(A.isDraw(h.winner)); assert.strictEqual(A.scoreB(h.winner), 0.5);
    assert.strictEqual(s.margin, null); assert.strictEqual(s.signed, null);
    for (const k of ["minWinnerP", "minWinnerScore", "decidedLeft", "raceDeficit"]) assert.strictEqual(s[k], null, k);
    assert(["repetition", "maxPlies"].includes(h.drawReason), h.drawReason);
    // the Brier score uses ½ as a draw's outcome
    const E = s.evs; if (E.length) assert(Math.abs(s.brier - E.reduce((a, e) => a + (e.p - 0.5) ** 2, 0) / E.length) < 1e-12);
  } else {
    assert(!A.isDraw(h.winner)); assert.strictEqual(A.scoreB(h.winner), re[0] === "B" ? 1 : 0);
    const m = parseFloat(re.slice(2));   // W+R / B+F (resignation, forfeit): no margin
    assert.strictEqual(s.margin, isFinite(m) ? Math.abs(m) : null);
    assert.strictEqual(h.drawReason, "");
    seen.decisive++;
  }
  // rules: komi, walls, repetition rule
  const km = /KM\[([^\]]*)\]/.exec(line); assert.strictEqual(h.komi, km && +km[1] !== 0 ? +km[1] : -0.5);
  const wb = /WB\[(\d+)\]/.exec(line), ww = /WW\[(\d+)\]/.exec(line);
  assert.strictEqual(h.wb, wb ? +wb[1] : 10); assert.strictEqual(h.ww, ww ? +ww[1] : 10);
  const ru = /RU\[([^\]]*)\]/.exec(line)[1], n = /repetitionDrawCount=(\d+)/.exec(ru);
  assert(h.ruleKnown); assert.strictEqual(h.repetition, n ? +n[1] : 0);
  assert.strictEqual(A.repLabel(h), n ? `rep ${n[1]}` : "rep off");
  // opening key: the first OPENING_PLIES moves
  assert.strictEqual(h.opening, g.moves.slice(0, A.OPENING_PLIES).map(m => m.raw).join(" "));
  assert.strictEqual(A.openingLabel(h.opening).split(" ").length, Math.min(A.OPENING_PLIES, g.moves.length));
  // repetition stretch vs an independent occurrence count: with the rule on (N), a decisive game never reaches N
  // occurrences of a position and a repetition draw ends on the N-th one, inside a repeated stretch
  const occ = occurrences(g), rs = A.repeatStretch(g);
  assert.strictEqual(rs.maxRun > 0, occ.max > 1);
  assert.strictEqual(s.cycleMax, rs.maxRun);
  if (h.repetition) {
    seen.ruleOn++; seen["n" + h.repetition] = (seen["n" + h.repetition] || 0) + 1;
    if (h.drawReason === "repetition") { assert.strictEqual(occ.last, h.repetition); assert(rs.endRun > 0 && rs.endStart >= 1); seen.repDraw++; }
    else assert(occ.max < h.repetition, `game ${i + 1}: ${occ.max} occurrences with N = ${h.repetition}`);
  } else { seen.ruleOff++; assert.notStrictEqual(h.drawReason, "repetition"); }
  if (h.drawReason === "maxPlies") seen.maxPliesDraw++;
  if (h.gtype === "fork") seen.fork++;
  if (h.wb < h.ww) seen.hcB++;
  if (h.ww < h.wb) seen.hcW++;
  if (h.komi !== -0.5) seen.komi++;
  if (h.pb !== h.pw) seen.match++;
  // full-search visits are this game's max v
  if (s.evs.length) assert.strictEqual(s.maxV, Math.max(...s.evs.map(e => e.v)));
});

// score with draws as ½
const S = lines.map((l, i) => A.summarize(l, A.headerOf(l, i)));
const r = A.bdw(S);
assert.strictEqual(r.n, S.length);
assert(Math.abs(r.score - (r.b + r.d / 2) / r.n) < 1e-12);
assert(Math.abs(r.drawRate - r.d / r.n) < 1e-12);
const t = A.bdw([{ winner: "B" }, { winner: "0" }, { winner: "W" }, { winner: "0" }, { winner: "?" }]);
assert.deepStrictEqual([t.b, t.d, t.w, t.n, t.score], [1, 2, 1, 4, 0.5]);

if (!process.argv[2]) {   // the fixture must cover every case listed above
  for (const [k, v] of Object.entries(seen)) assert(v > 0, `fixture has no ${k} game`);
}
console.log(`ok: ${lines.length} games`, JSON.stringify(seen), `B ${r.b} D ${r.d} W ${r.w} Black score ${(100 * r.score).toFixed(1)}%`);
