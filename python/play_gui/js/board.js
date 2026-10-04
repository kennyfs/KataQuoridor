"use strict";
// ---- Board drawing and hit testing (no game rules here: legality always comes from the engine) ----------
// Cells are (c, r) with c = 0..8 for columns a..i and r = 0..8 for rows 1..9. A wall is its anchor cell
// (c, r), c,r = 0..7, plus "h" (between rows r and r+1 under columns c, c+1) or "v" (between columns c and
// c+1 beside rows r, r+1). The view is a 180° rotation away from the standard diagram when Black is at the
// bottom: bottom = "w" draws row 1 at the bottom and column a on the left, bottom = "b" row 9 at the bottom
// and column a on the right, as a player sitting on that side sees the real board.
const N = 9, CELL = 50, GAP = 12, PITCH = CELL + GAP, PAD = 32;
const BOARD_PX = N * CELL + (N - 1) * GAP, SIZE = BOARD_PX + 2 * PAD;
const WALL_BAND = GAP / 2 + 10;   // a groove "captures" the pointer this far from its centre line
const SVGNS = "http://www.w3.org/2000/svg";
const COLS = "abcdefghi";
const ACC = { b: "var(--b-acc)", w: "var(--w-acc)" };
// Candidate colours as in Ogatak: the engine's best move bright light blue, the others green, fading with
// fewer visits.
const CAND_TOP = "#77eeff", CAND_OFF = "#99dd55";
function candAlpha(visits, top) {
  if (!top) return 0.5;
  return 0.25 + 0.75 * Math.sqrt(Math.max(0, Math.min(1, visits / top)));
}

function el(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

const cellName = (c, r) => COLS[c] + (r + 1);
function parseMove(m) {
  const c = COLS.indexOf(m[0]), r = parseInt(m.slice(1), 10) - 1;
  const o = m[m.length - 1];
  return o === "h" || o === "v" ? { wall: true, c, r, o } : { wall: false, c, r };
}

class Board {
  constructor(svg) {
    this.svg = svg;
    this.bottom = "w";
    this.hover = null;
    this.onHover = null;
    svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
  }

  // board cell -> view grid position (0..8 from left / top)
  view(c, r) { return this.bottom === "w" ? [c, N - 1 - r] : [N - 1 - c, r]; }
  cellXY(c, r) { const [vx, vy] = this.view(c, r); return [PAD + vx * PITCH, PAD + vy * PITCH]; }
  center(c, r) { const [x, y] = this.cellXY(c, r); return [x + CELL / 2, y + CELL / 2]; }
  wallRect(c, r, o, inset = 0) {
    const [x1, y1] = this.center(c, r), [x2, y2] = this.center(c + 1, r + 1);
    const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2, long = 2 * CELL + GAP - 2 * inset, thick = GAP - 2;
    return o === "h" ? { x: cx - long / 2, y: cy - thick / 2, width: long, height: thick }
                     : { x: cx - thick / 2, y: cy - long / 2, width: thick, height: long };
  }

  // Pointer (SVG units) -> {kind: "wall", move} | {kind: "cell", move} | null.
  // Near a groove (within WALL_BAND of its centre line) the pointer means a wall whose centre is the groove
  // crossing nearest to it; the orientation is the groove's (at a crossing: the nearer groove). Elsewhere
  // it means the cell under it. Walls are only offered when `wallsAllowed`.
  hitTest(x, y, wallsAllowed) {
    const gx = x - PAD, gy = y - PAD;
    if (gx < -4 || gy < -4 || gx > BOARD_PX + 4 || gy > BOARD_PX + 4) return null;
    if (wallsAllowed) {
      const off = CELL + GAP / 2;
      const kx = Math.max(0, Math.min(N - 2, Math.round((gx - off) / PITCH)));
      const ky = Math.max(0, Math.min(N - 2, Math.round((gy - off) / PITCH)));
      const dx = Math.abs(gx - (kx * PITCH + off)), dy = Math.abs(gy - (ky * PITCH + off));
      const nearV = dx <= WALL_BAND, nearH = dy <= WALL_BAND;
      if (nearV || nearH) {
        // the four cells around crossing (kx, ky) in view space; the anchor is their lowest column and row
        const c = this.bottom === "w" ? kx : N - 2 - kx;
        const r = this.bottom === "w" ? N - 2 - ky : ky;
        let o = nearV && nearH ? (dx < dy ? "v" : "h") : nearV ? "v" : "h";
        // right at a crossing, keep the orientation already previewed there instead of flickering
        const h = this.hover;
        if (nearV && nearH && Math.abs(dx - dy) < 4 && h && h.kind === "wall" && h.move.slice(0, 2) === cellName(c, r)) o = h.move[2];
        return { kind: "wall", move: cellName(c, r) + o };
      }
    }
    const vx = Math.floor(gx / PITCH), vy = Math.floor(gy / PITCH);
    if (vx < 0 || vy < 0 || vx >= N || vy >= N) return null;
    if (gx - vx * PITCH > CELL || gy - vy * PITCH > CELL) return null;   // in a groove
    const c = this.bottom === "w" ? vx : N - 1 - vx, r = this.bottom === "w" ? N - 1 - vy : vy;
    return { kind: "cell", move: cellName(c, r) };
  }

  eventPoint(e) {
    const pt = this.svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const m = this.svg.getScreenCTM();
    return m ? pt.matrixTransform(m.inverse()) : { x: -1, y: -1 };
  }

  // v: {pos, lastMove: {color, move}|null, prevPawn, legal: Set|null, mover, cands: {color, moves}|null,
  //     pv: {color, moves}|null, selected: bool, interactive: bool}
  // cands.moves: candidate dicts (move, visits, winrate for the mover, order) already filtered and sorted;
  // pv: a line to show instead of the candidates.
  draw(v) {
    this.v = v;
    const svg = this.svg;
    svg.innerHTML = "";
    const defs = el("defs", {}, svg);
    const f = el("filter", { id: "shadow", x: "-30%", y: "-30%", width: "160%", height: "160%" }, defs);
    el("feDropShadow", { dx: 0, dy: 1.5, stdDeviation: 1.6, "flood-color": "#000", "flood-opacity": 0.35 }, f);
    const glow = el("filter", { id: "glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
    el("feDropShadow", { dx: 0, dy: 0, stdDeviation: 3.5, "flood-color": "#fff", "flood-opacity": 0.9 }, glow);

    el("rect", { x: 4, y: 4, width: SIZE - 8, height: SIZE - 8, rx: 22, fill: "var(--board)", stroke: "var(--board-edge)", "stroke-width": 3 }, svg);
    // coordinates on all four edges
    for (let i = 0; i < N; i++) {
      const [x] = this.center(i, 0), [, y] = this.center(0, i);
      for (const [tx, ty, s] of [[x, PAD - 11, COLS[i]], [x, SIZE - PAD + 20, COLS[i]], [PAD - 15, y + 4, i + 1], [SIZE - PAD + 15, y + 4, i + 1]]) {
        const t = el("text", { x: tx, y: ty, "text-anchor": "middle", "font-size": 12.5, "font-weight": 650, fill: "var(--label)" }, svg);
        t.textContent = s;
      }
    }
    // cells; goal rows tinted (Black heads to row 1, White to row 9)
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      const [x, y] = this.cellXY(c, r);
      const fill = r === 0 ? "var(--cell-goal-b)" : r === N - 1 ? "var(--cell-goal-w)" : "var(--cell)";
      el("rect", { x, y, width: CELL, height: CELL, rx: 8, fill }, svg);
    }
    if (!v || !v.pos) return;
    const pos = v.pos;

    // walls
    const last = v.lastMove;
    for (const w of pos.walls) {
      const m = parseMove(w.move), isLast = last && last.move === w.move;
      el("rect", { ...this.wallRect(m.c, m.r, m.o), rx: 4, fill: ACC[w.color],
        stroke: isLast ? "#fff" : "rgba(0,0,0,.25)", "stroke-width": isLast ? 2 : 1, filter: isLast ? "url(#glow)" : "url(#shadow)" }, svg);
    }

    // where the last pawn move came from
    if (last && !parseMove(last.move).wall && v.prevPawn) {
      const p = parseMove(v.prevPawn), [cx, cy] = this.center(p.c, p.r);
      el("circle", { cx, cy, r: CELL * 0.33, fill: "none", stroke: ACC[last.color], "stroke-width": 2, "stroke-dasharray": "4 4", opacity: 0.85 }, svg);
    }

    // legal pawn destinations (not where a candidate is drawn)
    const candCells = new Set(v.pv ? [] : ((v.cands && v.cands.moves) || []).map(m => m.move));
    if (v.interactive && v.legal) {
      for (const mv of v.legal) {
        if (mv.length !== 2 || candCells.has(mv)) continue;
        const m = parseMove(mv), [cx, cy] = this.center(m.c, m.r);
        el("circle", { cx, cy, r: v.selected ? 9 : 7, fill: ACC[v.mover], opacity: v.selected ? 0.85 : 0.55, class: "dest" }, svg);
      }
    }

    // pawns
    for (const p of ["b", "w"]) {
      const m = parseMove(pos.pawns[p]), [cx, cy] = this.center(m.c, m.r);
      const isLast = last && last.color === p && !parseMove(last.move).wall;
      if (isLast) el("circle", { cx, cy, r: CELL * 0.45, fill: "none", stroke: ACC[p], "stroke-width": 2.5, opacity: 0.6 }, svg);
      if (v.interactive && v.selected && p === v.mover) el("circle", { cx, cy, r: CELL * 0.47, fill: "none", stroke: "var(--sel-ring)", "stroke-width": 3 }, svg);
      el("circle", { cx, cy, r: CELL * 0.34, fill: p === "b" ? "var(--b-pawn)" : "var(--w-pawn)", stroke: ACC[p], "stroke-width": 4, filter: "url(#shadow)" }, svg);
      el("circle", { cx: cx - CELL * 0.1, cy: cy - CELL * 0.11, r: CELL * 0.09, fill: "#fff", opacity: p === "b" ? 0.18 : 0.8 }, svg);
    }

    // candidates or a principal variation, above everything else
    const top = el("g", { "pointer-events": "none" }, svg);
    if (v.pv && v.pv.moves.length) this.drawPV(top, v.pv);
    else if (v.cands) this.drawCands(top, v.cands);

    this.previewLayer = el("g", { "pointer-events": "none" }, svg);
    this.drawPreview();
  }

  drawCands(g, cands) {
    const list = cands.moves;
    if (!list.length) return;
    const topVisits = Math.max(...list.map(m => m.visits || 0));
    // walls first so the pawn-move discs stay readable; the best move last, on top
    const ordered = [...list].sort((a, b) => (b.move.length - a.move.length) || (b.order - a.order));
    for (const m of ordered) {
      const p = parseMove(m.move), best = m.order === 0;
      const col = best ? CAND_TOP : CAND_OFF, a = best ? 1 : candAlpha(m.visits, topVisits);
      const win = m.winrate == null ? "" : (100 * m.winrate).toFixed(0);
      if (p.wall) {
        const r = this.wallRect(p.c, p.r, p.o);
        el("rect", { ...r, rx: 4, fill: col, "fill-opacity": a, stroke: best ? "#0b6c7c" : "#3d6b1c", "stroke-opacity": Math.min(1, a + 0.2), "stroke-width": 1.5 }, g);
        const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
        el("rect", { x: cx - 15, y: cy - 8, width: 30, height: 16, rx: 8, fill: col, "fill-opacity": Math.max(a, 0.85), stroke: "#14303a", "stroke-opacity": 0.5, "stroke-width": 1 }, g);
        this.text(g, cx, cy + 4, win, 10.5, 750, "#10242c");
      } else {
        const [cx, cy] = this.center(p.c, p.r);
        el("circle", { cx, cy, r: CELL * 0.44, fill: col, "fill-opacity": a, stroke: best ? "#0b6c7c" : "#3d6b1c", "stroke-opacity": Math.min(1, a + 0.2), "stroke-width": 1.5 }, g);
        this.text(g, cx, cy + 1, win, 15, 750, "#10242c");
        this.text(g, cx, cy + 13, fmtVisits(m.visits), 9.5, 600, "#10242c");
      }
    }
  }

  drawPV(g, pv) {
    let c = pv.color;
    pv.moves.slice(0, 12).forEach((mv, i) => {
      const p = parseMove(mv);
      if (p.wall) {
        const r = this.wallRect(p.c, p.r, p.o);
        el("rect", { ...r, rx: 4, fill: ACC[c], opacity: 0.8, stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "4 3" }, g);
        const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
        el("circle", { cx, cy, r: 9, fill: c === "b" ? "var(--b-pawn)" : "var(--w-pawn)", stroke: ACC[c], "stroke-width": 2 }, g);
        this.text(g, cx, cy + 4, i + 1, 11, 750, c === "b" ? "#fff" : "#111");
      } else {
        const [cx, cy] = this.center(p.c, p.r);
        el("circle", { cx, cy, r: CELL * 0.3, fill: c === "b" ? "var(--b-pawn)" : "var(--w-pawn)", opacity: 0.85, stroke: ACC[c], "stroke-width": 3 }, g);
        this.text(g, cx, cy + 5, i + 1, 14, 750, c === "b" ? "#fff" : "#111");
      }
      c = c === "b" ? "w" : "b";
    });
  }

  text(g, x, y, s, size, weight, fill) {
    const t = el("text", { x, y, "text-anchor": "middle", "font-size": size, "font-weight": weight, fill }, g);
    t.textContent = s;
    return t;
  }

  // hover: {kind, move, legal} or null
  setHover(h) {
    const same = (a, b) => a === b || (a && b && a.kind === b.kind && a.move === b.move && a.legal === b.legal);
    if (same(h, this.hover)) return;
    this.hover = h;
    this.drawPreview();
  }

  drawPreview() {
    const g = this.previewLayer;
    if (!g) return;
    g.innerHTML = "";
    const h = this.hover, v = this.v;
    if (!h || !v || !v.interactive) return;
    const m = parseMove(h.move);
    if (h.kind === "wall") {
      el("rect", { ...this.wallRect(m.c, m.r, m.o), rx: 4, fill: h.legal ? ACC[v.mover] : "var(--bad)", opacity: h.legal ? 0.6 : 0.5,
        stroke: h.legal ? "#fff" : "var(--bad)", "stroke-width": 1.5, class: h.legal ? "pv-ok" : "pv-bad" }, g);
    } else if (h.legal) {
      const [cx, cy] = this.center(m.c, m.r);
      el("circle", { cx, cy, r: CELL * 0.34, fill: v.mover === "b" ? "var(--b-pawn)" : "var(--w-pawn)", stroke: ACC[v.mover], "stroke-width": 4, opacity: 0.5 }, g);
    }
  }
}

function fmtVisits(n) {
  if (n == null) return "";
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
  return String(n);
}
