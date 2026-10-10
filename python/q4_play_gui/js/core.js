/* Pure helpers of the Q4 play GUI: notation, the board's rotation and wall grooves, input parsing, formatting.
 * No DOM here, so tests/core_check.js can run it under Node. Coordinates: cell (x, y), x = column a..k (0..10),
 * y = row 1..11 (0..10), y up. Wall anchor (ax, ay), ax, ay in 0..9: horizontal "h" lies on the line y = ay + 1
 * over columns ax..ax+2, vertical "v" on the line x = ax + 1 over rows ay..ay+2 (continuous coordinates, cell x
 * spans [x, x + 1]). The view is rotated by `rot` quarter turns counter-clockwise: seat S (bottom) is rot 0,
 * W (left) rot 1, N (top) rot 2, E (right) rot 3. */
const Q4Core = (function () {
  const N = 11;                                   // board size
  const SEAT_LETTERS = ["S", "W", "N", "E"];
  const SEAT_NAMES = ["South", "West", "North", "East"];
  const SEAT_COLORS = ["#0072b2", "#e69f00", "#cc79a7", "#009e73"];   // the Q4 viewer's Okabe-Ito set
  const DRAW_COLOR = "#9aa5b1";
  const CENTER = [5, 5];                          // f6
  const PAWN_RE = /^[a-k](?:1[01]|[1-9])$/;
  const WALL_RE = /^[a-j](?:10|[1-9])[hv]$/;

  function cellName(x, y) { return String.fromCharCode(97 + x) + String(y + 1); }

  // "f2" -> {kind:"pawn", x, y}; "e5h" -> {kind:"wall", ax, ay, h}; anything else -> null
  function parseMove(text) {
    const t = String(text || "").trim().toLowerCase();
    if (PAWN_RE.test(t)) return { kind: "pawn", x: t.charCodeAt(0) - 97, y: parseInt(t.slice(1), 10) - 1 };
    if (WALL_RE.test(t)) {
      return { kind: "wall", ax: t.charCodeAt(0) - 97, ay: parseInt(t.slice(1, -1), 10) - 1, h: t.endsWith("h") };
    }
    return null;
  }

  function wallName(ax, ay, h) { return String.fromCharCode(97 + ax) + String(ay + 1) + (h ? "h" : "v"); }

  // Input of the move box: "f2", "E5H", "j10v", "  e5h " -> the canonical text, or {error}.
  function normaliseMoveInput(text) {
    const t = String(text || "").trim().toLowerCase();
    if (!t) return { error: "type a move, e.g. f2 or e5h" };
    if (parseMove(t)) return { move: t };
    return { error: "not a move: " + text + " (pawn f2, wall e5h or j10v)" };
  }

  // Rotation of the continuous board: one quarter turn counter-clockwise about the centre (5.5, 5.5).
  function rotatePoint(px, py) { return [N - py, px]; }
  function unrotatePoint(px, py) { return [py, N - px]; }
  function rotateCellPoint(x, y, rot) {
    let p = [x + 0.5, y + 0.5];
    for (let k = 0; k < ((rot % 4) + 4) % 4; k++) p = rotatePoint(p[0], p[1]);
    return p;
  }
  // Inverse for a continuous point in the view (before the screen flip): the base board point.
  function viewToBase(px, py, rot) {
    let p = [px, py];
    for (let k = 0; k < ((rot % 4) + 4) % 4; k++) p = unrotatePoint(p[0], p[1]);
    return p;
  }
  // The cell at a base-board cell index (for drawing the cell in view order: row 0 is the top of the view).
  function cellAtView(col, row, rot) {
    const b = viewToBase(col + 0.5, N - row - 0.5, rot);
    return { x: Math.floor(b[0]), y: Math.floor(b[1]) };
  }

  // The two end points of a wall in base continuous coordinates.
  function wallSegment(ax, ay, h) {
    if (h) return [[ax, ay + 1], [ax + 2, ay + 1]];
    return [[ax + 1, ay], [ax + 1, ay + 2]];
  }

  // The wall groove under a continuous base point: which wall (anchor and orientation) the pointer is on. A groove is
  // a band of GROOVE_BAND cells either side of a grid line (centres at the lattice corners); anywhere else the pointer
  // is on a cell and this returns null. Anchors 0..9 => corners 1..10, so the outer lines have no walls.
  const GROOVE_BAND = 0.3;
  function grooveAt(bx, by) {
    const i = Math.round(bx), j = Math.round(by);
    if (i < 1 || i > 10 || j < 1 || j > 10) return null;
    const dh = Math.abs(by - j), dv = Math.abs(bx - i);
    if (Math.min(dh, dv) >= GROOVE_BAND) return null;
    return { ax: i - 1, ay: j - 1, h: dh <= dv };
  }

  // A rectangle (in base units) for a wall, for drawing: the wall is 2 cells long and a thin bar.
  function wallRect(ax, ay, h, thick) {
    const t = thick || 0.12;
    if (h) return { x: ax, y: ay + 1 - t / 2, w: 2, h: t };
    return { x: ax + 1 - t / 2, y: ay, w: t, h: 2 };
  }

  // Map seat s to the rotation that puts it at the bottom of the view.
  function rotationForSeat(seat) { return seat; }

  // Probability text: 0.1234 -> "12.3%".
  function pct(p, digits) {
    if (p === null || p === undefined || Number.isNaN(p)) return "–";
    return (100 * p).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  // Segments of a stacked bar over [S W N E draw] as fractions that sum to 1 (negative or NaN count as 0).
  function barSegments(values) {
    const v = values.map(x => (Number.isFinite(x) && x > 0 ? x : 0));
    const total = v.reduce((a, b) => a + b, 0) || 1;
    return v.map(x => x / total);
  }

  // The seat of each wall on the board: walls are listed in placement order by the event list.
  function wallSeats(events) {
    const seats = {};
    for (const e of events) {
      if (e.applied === false || e.elim || !e.a) continue;
      if (e.a.endsWith("h") || e.a.endsWith("v")) seats[e.a] = e.seat;
    }
    return seats;
  }

  // Sorting for the candidate table. key: "lcb" | "visits" | "prior" | "utility"; higher first.
  function sortCandidates(list, key) {
    return list.slice().sort((a, b) => (b[key] - a[key]) || (b.visits - a.visits) || (a.move < b.move ? -1 : 1));
  }

  // The mover's win probability of a candidate: value[seat].
  function moverWin(candidate, seat) { return candidate.value ? candidate.value[seat] : null; }

  return {
    N, SEAT_LETTERS, SEAT_NAMES, SEAT_COLORS, DRAW_COLOR, CENTER,
    cellName, parseMove, wallName, normaliseMoveInput,
    rotatePoint, unrotatePoint, rotateCellPoint, viewToBase, cellAtView,
    wallSegment, grooveAt, wallRect, rotationForSeat,
    pct, barSegments, wallSeats, sortCandidates, moverWin,
  };
})();

if (typeof module !== "undefined" && module.exports) module.exports = Q4Core;
if (typeof window !== "undefined") window.Q4Core = Q4Core;
