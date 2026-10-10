// Node check of the pure helpers of the play GUI: `node python/q4_play_gui/tests/core_check.js`.
const assert = require("assert");
const C = require("../js/core.js");

// notation
assert.strictEqual(C.cellName(5, 0), "f1");
assert.strictEqual(C.cellName(10, 10), "k11");
assert.deepStrictEqual(C.parseMove("f2"), { kind: "pawn", x: 5, y: 1 });
assert.deepStrictEqual(C.parseMove("e5h"), { kind: "wall", ax: 4, ay: 4, h: true });
assert.deepStrictEqual(C.parseMove("j10v"), { kind: "wall", ax: 9, ay: 9, h: false });
assert.strictEqual(C.parseMove("k12"), null);
assert.strictEqual(C.parseMove("a0h"), null);
assert.strictEqual(C.wallName(4, 4, true), "e5h");
assert.strictEqual(C.normaliseMoveInput("  E5H ").move, "e5h");
assert.ok(C.normaliseMoveInput("zz").error);

// rotation: seat S at the bottom for rot 0; each quarter turn brings the next seat to the bottom
const bottomEdgeOf = (rot) => {
  // the base edge that ends at the bottom (y_view = 11) after rot quarter turns
  const pts = [[5.5, 0.0], [0.0, 5.5], [5.5, 11.0], [11.0, 5.5]];   // S, W, N, E edge midpoints (base)
  let best = null;
  for (let s = 0; s < 4; s++) {
    let p = pts[s];
    for (let k = 0; k < rot; k++) p = C.rotatePoint(p[0], p[1]);
    if (Math.abs(p[1] - 0.0) < 1e-9 || Math.abs(p[1] - 11.0) < 1e-9) best = best; // placeholder
    if (Math.abs(p[0] - 5.5) < 1e-9 && p[1] < 1e-9) best = s;
  }
  return best;
};
assert.strictEqual(bottomEdgeOf(0), 0, "rot 0: S at the bottom");
assert.strictEqual(bottomEdgeOf(1), 1, "rot 1: W at the bottom");
assert.strictEqual(bottomEdgeOf(2), 2, "rot 2: N at the bottom");
assert.strictEqual(bottomEdgeOf(3), 3, "rot 3: E at the bottom");

// rotation round trip and cell placement
for (let rot = 0; rot < 4; rot++) {
  for (let x = 0; x < 11; x++) for (let y = 0; y < 11; y++) {
    const [px, py] = C.rotateCellPoint(x, y, rot);
    const b = C.viewToBase(px, py, rot);
    assert.ok(Math.abs(b[0] - (x + 0.5)) < 1e-9 && Math.abs(b[1] - (y + 0.5)) < 1e-9, "round trip");
    const cell = C.cellAtView(Math.floor(px), 10 - Math.floor(py), rot);
    // the view cell (col, row) in screen coordinates maps back to the same base cell
    assert.deepStrictEqual(cell, { x, y }, `cell ${x},${y} rot ${rot}`);
  }
}

// grooves: the centre of each horizontal / vertical wall maps back to that wall, for every anchor
for (let ax = 0; ax < 10; ax++) for (let ay = 0; ay < 10; ay++) {
  const hc = [ax + 1, ay + 1];
  assert.deepStrictEqual(C.grooveAt(hc[0] + 0.2, hc[1] + 0.05), { ax, ay, h: true }, `h ${ax},${ay}`);
  assert.deepStrictEqual(C.grooveAt(hc[0] + 0.05, hc[1] + 0.2), { ax, ay, h: false }, `v ${ax},${ay}`);
  const seg = C.wallSegment(ax, ay, true);
  assert.deepStrictEqual(seg, [[ax, ay + 1], [ax + 2, ay + 1]]);
}
assert.strictEqual(C.grooveAt(0.2, 5), null);
assert.strictEqual(C.grooveAt(11.4, 5), null);

// wall rectangles stay inside the board
for (let ax = 0; ax < 10; ax++) for (let ay = 0; ay < 10; ay++) {
  for (const h of [true, false]) {
    const r = C.wallRect(ax, ay, h);
    assert.ok(r.x >= 0 && r.y >= 0 && r.x + r.w <= 11 && r.y + r.h <= 11);
  }
}

// formatting and bars
assert.strictEqual(C.pct(0.1234), "12.3%");
assert.strictEqual(C.pct(null), "–");
const bar = C.barSegments([1, 1, 0, 2, -1]);
assert.ok(Math.abs(bar.reduce((a, b) => a + b, 0) - 1) < 1e-12);
assert.strictEqual(bar[2], 0);

// the seat of each wall from the event list (undone events do not count)
const seats = C.wallSeats([{ seat: 0, a: "f2", applied: true }, { seat: 2, a: "e5h", applied: true },
                           { seat: 1, a: "a1v", applied: false }, { seat: 3, elim: true, applied: true }]);
assert.deepStrictEqual(seats, { e5h: 2 });

// candidates: sorted by lcb, then visits
const sorted = C.sortCandidates([{ move: "f2", lcb: -1, visits: 5 }, { move: "e2", lcb: 0.5, visits: 2 }], "lcb");
assert.strictEqual(sorted[0].move, "e2");
console.log("core_check: ok");

// a point in a cell centre is not a groove; a point on a grid line is
assert.strictEqual(C.grooveAt(3.5, 3.5), null);
assert.deepStrictEqual(C.grooveAt(3.5, 4.0), { ax: 3, ay: 3, h: true }); // on the line y = 4, between x = 3 and 4
assert.ok(C.grooveAt(3.5, 4.05) !== null);
console.log("core_check: grooves ok");
