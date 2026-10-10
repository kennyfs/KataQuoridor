/* The 11 x 11 Q4 board as SVG, drawn like the Duel board (python/play_gui/js/board.js): rounded cells with grooves
 * between them, walls in the grooves, pawns with a highlight, candidate discs / wall bars with the mover's win chance
 * (and visits), the principal variation of a hovered candidate, and a preview of the wall / pawn move under the
 * pointer (legal in the seat colour, illegal red and not clickable). The only text is the rank and file letters
 * outside the edges, the seat letter in a pawn, and the numbers of the candidates and the variation: no cell and
 * no move name is written on the board. Rotation is set by the app (see core.js); the board knows no rules.
 * Depends on Q4Core (core.js) only; no state of its own beyond the last view it was given. */
(function (root) {
  const C = root.Q4Core || require("./core.js");
  const N = C.N, CELL = 46, GAP = 11, PITCH = CELL + GAP, PAD = 30;
  const SIZE = N * CELL + (N - 1) * GAP + 2 * PAD;
  const SVGNS = "http://www.w3.org/2000/svg";
  // candidate colours as in the Duel board (Ogatak): the best bright light blue, the others green fading with visits
  const CAND_TOP = "#77eeff", CAND_OFF = "#99dd55";
  const COLS = "abcdefghijk";

  function el(name, attrs, parent) {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function candAlpha(visits, top) {
    if (!top) return 0.5;
    return 0.25 + 0.75 * Math.sqrt(Math.max(0, Math.min(1, visits / top)));
  }

  function fmtVisits(n) {
    if (n == null) return "";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + "k";
    return String(n);
  }

  class Q4Board {
    // handlers: onMove(name) a legal move was clicked, onCell(name) a cell without a legal pawn move was clicked,
    // onIllegal(name) an illegal wall was clicked, onHover(hit|null) the pointer target changed
    constructor(svg, handlers) {
      this.svg = svg;
      this.h = handlers || {};
      this.rot = 0;
      this.view = null;
      this.hover = null;   // {kind:"cell"|"wall", name, ...} under the pointer
      this.down = null;
      svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
      svg.addEventListener("pointermove", (e) => { if (e.pointerType !== "touch") this._move(e); });
      svg.addEventListener("pointerleave", () => this._setHover(null));
      // pointerdown + pointerup instead of click: the board is redrawn several times a second while the analysis
      // runs, and a click whose target was replaced between press and release is never delivered
      svg.addEventListener("pointerdown", (e) => { const hv = e.button === 0 ? this._pick(e) : null; this.down = hv ? hv.name : null; });
      svg.addEventListener("pointerup", (e) => this._up(e));
    }

    setRotation(rot) { this.rot = ((rot % 4) + 4) % 4; if (this.view) this.render(this.view); }

    // The screen position (SVG units) of a base point in continuous coordinates, in the current rotation.
    // Grid line k (a groove) is at PAD + k * PITCH - GAP / 2, so that cell x has its centre at x + 0.5.
    _pt(bx, by) {
      let p = [bx, by];
      for (let k = 0; k < this.rot; k++) p = C.rotatePoint(p[0], p[1]);
      return [PAD - GAP / 2 + p[0] * PITCH, PAD - GAP / 2 + (N - p[1]) * PITCH];
    }
    _center(x, y) { return this._pt(x + 0.5, y + 0.5); }

    // The rectangle of a wall (in the groove, 2 cells + the groove between them long), in screen units.
    _wallBox(ax, ay, h) {
      const seg = C.wallSegment(ax, ay, h);
      const a = this._pt(seg[0][0], seg[0][1]), b = this._pt(seg[1][0], seg[1][1]);
      const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2, thick = GAP - 2;
      const long = Math.max(Math.abs(a[0] - b[0]), Math.abs(a[1] - b[1])) - GAP;
      return Math.abs(a[0] - b[0]) > Math.abs(a[1] - b[1])
        ? { x: cx - long / 2, y: cy - thick / 2, width: long, height: thick }
        : { x: cx - thick / 2, y: cy - long / 2, width: thick, height: long };
    }

    _svgPoint(e) {
      const pt = this.svg.createSVGPoint();
      pt.x = e.clientX; pt.y = e.clientY;
      const m = this.svg.getScreenCTM();
      return m ? pt.matrixTransform(m.inverse()) : { x: -1, y: -1 };
    }

    // Screen event -> base point in continuous coordinates (before rotation).
    _base(e) {
      const p = this._svgPoint(e);
      const vx = (p.x - PAD + GAP / 2) / PITCH, vy = N - (p.y - PAD + GAP / 2) / PITCH;
      return C.viewToBase(vx, vy, this.rot);
    }

    // What the pointer is on: a wall groove, a cell, or nothing (computed from the event itself).
    _pick(e) {
      const v = this.view;
      if (!v || v.inactive) return null;
      const b = this._base(e);
      // A pawn destination (a dot, or the cell of a legal pawn move) wins over a groove next to it.
      const cx = Math.floor(b[0]), cy = Math.floor(b[1]);
      if (cx >= 0 && cx < N && cy >= 0 && cy < N) {
        const name = C.cellName(cx, cy);
        if (v.legalPawns.has(name) && Math.hypot(b[0] - (cx + 0.5), b[1] - (cy + 0.5)) < 0.4) return { kind: "cell", name, x: cx, y: cy, legal: true };
      }
      if (v.wallsAllowed && b[0] >= 0 && b[0] <= N && b[1] >= 0 && b[1] <= N) {
        const g = C.grooveAt(b[0], b[1]);
        if (g) {
          const name = C.wallName(g.ax, g.ay, g.h);
          return { kind: "wall", name, ax: g.ax, ay: g.ay, h: g.h, legal: v.legalWalls.has(name) };
        }
      }
      if (cx >= 0 && cx < N && cy >= 0 && cy < N) {
        const name = C.cellName(cx, cy);
        return { kind: "cell", name, x: cx, y: cy, legal: v.legalPawns.has(name) };
      }
      return null;
    }

    _move(e) {
      if (!this.view) return;
      this._setHover(this._pick(e));
    }

    _setHover(next) {
      const same = (a, b) => a === b || (a && b && a.kind === b.kind && a.name === b.name && a.legal === b.legal);
      this.svg.style.cursor = !next ? "" : next.legal ? "pointer" : next.kind === "wall" ? "not-allowed" : "";
      if (same(next, this.hover)) return;
      this.hover = next;
      this._drawHover();
      if (this.h.onHover) this.h.onHover(next);
    }

    _up(e) {
      const down = this.down;
      this.down = null;
      if (!this.view || e.button !== 0) return;
      const hv = this._pick(e);
      if (!hv || hv.name !== down) return;
      // On touch screens the first tap on a wall only previews it; a second tap on the same wall places it.
      if (e.pointerType === "touch" && hv.kind === "wall" && (!this.hover || this.hover.name !== hv.name)) { this._setHover(hv); return; }
      if (hv.kind === "wall") {
        if (hv.legal && this.h.onMove) this.h.onMove(hv.name);
        else if (this.h.onIllegal) this.h.onIllegal(hv.name);
        return;
      }
      if (hv.legal && this.h.onMove) return this.h.onMove(hv.name);
      if (this.h.onCell) this.h.onCell(hv.name);
    }

    render(view) {
      this.view = view;
      const svg = this.svg;
      svg.textContent = "";
      const defs = el("defs", {}, svg);
      const f = el("filter", { id: "q4shadow", x: "-30%", y: "-30%", width: "160%", height: "160%" }, defs);
      el("feDropShadow", { dx: 0, dy: 1.5, stdDeviation: 1.6, "flood-color": "#000", "flood-opacity": 0.35 }, f);
      const glow = el("filter", { id: "q4glow", x: "-50%", y: "-50%", width: "200%", height: "200%" }, defs);
      el("feDropShadow", { dx: 0, dy: 0, stdDeviation: 3.5, "flood-color": "#fff", "flood-opacity": 0.9 }, glow);
      el("rect", { x: 4, y: 4, width: SIZE - 8, height: SIZE - 8, rx: 22, fill: "var(--board)", stroke: "var(--board-edge)", "stroke-width": 3 }, svg);
      this._edges(svg);
      this._cells(svg);
      if (view && view.position) {
        this._walls(svg);
        this._last(svg);
        this._legal(svg);
        this._pawns(svg);
        const top = el("g", { "pointer-events": "none" }, svg);
        if (view.pv && view.pv.length) this._pv(top);
        else this._candidates(top);
      }
      this.previewLayer = el("g", { "pointer-events": "none" }, svg);
      if (this.hover && view) this.hover.legal = this.hover.kind === "wall" ? view.legalWalls.has(this.hover.name) : view.legalPawns.has(this.hover.name);
      this._drawHover();
    }

    // rank and file letters outside the four edges, in the order the rotated view shows them
    _edges(svg) {
      const even = this.rot % 2 === 0;
      const file = (cell) => even ? COLS[cell.x] : String(cell.y + 1);   // along the top / bottom edge
      const rank = (cell) => even ? String(cell.y + 1) : COLS[cell.x];   // along the left / right edge
      for (let i = 0; i < N; i++) {
        const mid = PAD + i * PITCH + CELL / 2;
        const top = C.cellAtView(i, 0, this.rot), bottom = C.cellAtView(i, N - 1, this.rot);
        const left = C.cellAtView(0, i, this.rot), right = C.cellAtView(N - 1, i, this.rot);
        for (const [tx, ty, s] of [[mid, PAD - 10, file(top)], [mid, SIZE - PAD + 20, file(bottom)],
                                   [PAD - 14, mid + 4, rank(left)], [SIZE - PAD + 14, mid + 4, rank(right)]]) {
          this._text(this.svg, tx, ty, s, 12, 650, "var(--label)");
        }
      }
    }

    _cells(svg) {
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
        const x = PAD + i * PITCH, y = PAD + j * PITCH;
        const b = C.viewToBase(i + 0.5, N - j - 0.5, this.rot);
        const centre = Math.floor(b[0]) === C.CENTER[0] && Math.floor(b[1]) === C.CENTER[1];
        el("rect", { x, y, width: CELL, height: CELL, rx: 8, fill: centre ? "var(--cell-goal)" : "var(--cell)" }, svg);
        if (centre) el("circle", { cx: x + CELL / 2, cy: y + CELL / 2, r: 5, fill: "none", stroke: "#b8860b", "stroke-width": 1.8 }, svg);
      }
    }

    _walls(svg) {
      const v = this.view, pos = v.position, owner = v.wallSeats || {};
      const last = v.lastMove && v.lastMove.kind === "wall" ? C.wallName(v.lastMove.ax, v.lastMove.ay, v.lastMove.h) : null;
      for (const w of pos.walls || []) {
        const m = C.parseMove(w);
        if (!m) continue;
        const seat = owner[w], dead = seat !== undefined && !pos.alive[seat], isLast = last === w;
        el("rect", Object.assign(this._wallBox(m.ax, m.ay, m.h), { rx: 4,
          fill: seat === undefined ? "#777" : v.colors[seat], "fill-opacity": dead ? 0.35 : 1,
          stroke: isLast ? "#fff" : "rgba(0,0,0,.25)", "stroke-width": isLast ? 2 : 1,
          filter: isLast ? "url(#q4glow)" : "url(#q4shadow)" }), svg);
      }
    }

    // where the last pawn move came from
    _last(svg) {
      const v = this.view;
      if (!v.lastMove || v.lastMove.kind !== "pawn" || !v.prevCell) return;
      const p = C.parseMove(v.prevCell);
      if (!p) return;
      const [cx, cy] = this._center(p.x, p.y);
      el("circle", { cx, cy, r: CELL * 0.33, fill: "none", stroke: v.colors[v.lastSeat], "stroke-width": 2,
        "stroke-dasharray": "4 4", opacity: 0.85 }, svg);
    }

    // legal pawn destinations of the shown seat (not where a candidate is drawn)
    _legal(svg) {
      const v = this.view;
      if (!v.legalDots || v.inactive) return;
      const skip = new Set(v.pv && v.pv.length ? [] : (v.candidates || []).map(c => c.move));
      for (const name of v.legalDots) {
        const p = C.parseMove(name);
        if (!p || p.kind !== "pawn" || skip.has(name)) continue;
        const [cx, cy] = this._center(p.x, p.y);
        el("circle", { cx, cy, r: v.dotSelected ? 9 : 7, fill: v.colors[v.dotSeat], opacity: v.dotSelected ? 0.85 : 0.55 }, svg);
      }
    }

    _pawns(svg) {
      const v = this.view, pos = v.position;
      for (let s = 0; s < 4; s++) {
        if (!pos.alive[s] || !pos.pawns[s]) continue;
        const p = C.parseMove(pos.pawns[s]);
        if (!p) continue;
        const [cx, cy] = this._center(p.x, p.y);
        const isLast = v.lastMove && v.lastMove.kind === "pawn" && v.lastSeat === s;
        if (isLast) el("circle", { cx, cy, r: CELL * 0.45, fill: "none", stroke: v.colors[s], "stroke-width": 2.5, opacity: 0.6 }, svg);
        if (s === v.dotSeat && v.dotSelected) el("circle", { cx, cy, r: CELL * 0.47, fill: "none", stroke: "var(--sel-ring)", "stroke-width": 3 }, svg);
        if (s === pos.toMove && !v.inactive) el("circle", { cx, cy, r: CELL * 0.42, fill: "none", stroke: "#fff", "stroke-width": 2, "stroke-dasharray": "2 3", opacity: 0.9 }, svg);
        el("circle", { cx, cy, r: CELL * 0.34, fill: v.colors[s], stroke: "#fff", "stroke-width": 3, filter: "url(#q4shadow)" }, svg);
        el("circle", { cx: cx - CELL * 0.1, cy: cy - CELL * 0.11, r: CELL * 0.08, fill: "#fff", opacity: 0.55 }, svg);
        this._text(svg, cx + 1, cy + 5, C.SEAT_LETTERS[s], 14, 800, "#fff");
      }
    }

    _candidates(g) {
      const list = this.view.candidates;
      if (!list || !list.length) return;
      const topVisits = Math.max(...list.map(c => c.visits || 0));
      // walls first so the pawn-move discs stay readable; the best move (index 0) last, on top
      const isWall = (c) => { const p = C.parseMove(c.move); return p && p.kind === "wall"; };
      const ordered = list.map((c, i) => ({ c, i })).sort((a, b) => (isWall(b.c) - isWall(a.c)) || (b.i - a.i));
      for (const { c, i } of ordered) {
        const p = C.parseMove(c.move);
        if (!p) continue;
        const best = i === 0, col = best ? CAND_TOP : CAND_OFF, a = best ? 1 : candAlpha(c.visits, topVisits);
        const win = c.win == null ? "" : (100 * c.win).toFixed(0);
        const stroke = best ? "#0b6c7c" : "#3d6b1c";
        if (p.kind === "wall") {
          const r = this._wallBox(p.ax, p.ay, p.h);
          el("rect", Object.assign({}, r, { rx: 4, fill: col, "fill-opacity": a, stroke, "stroke-opacity": Math.min(1, a + 0.2), "stroke-width": 1.5 }), g);
          const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
          el("rect", { x: cx - 15, y: cy - 8, width: 30, height: 16, rx: 8, fill: col, "fill-opacity": Math.max(a, 0.85), stroke: "#14303a", "stroke-opacity": 0.5, "stroke-width": 1 }, g);
          this._text(g, cx, cy + 4, win, 10.5, 750, "#10242c");
        } else {
          const [cx, cy] = this._center(p.x, p.y);
          el("circle", { cx, cy, r: CELL * 0.44, fill: col, "fill-opacity": a, stroke, "stroke-opacity": Math.min(1, a + 0.2), "stroke-width": 1.5 }, g);
          this._text(g, cx, cy + 1, win, 14, 750, "#10242c");
          this._text(g, cx, cy + 12, fmtVisits(c.visits), 9, 600, "#10242c");
        }
      }
    }

    _pv(g) {
      const v = this.view;
      v.pv.slice(0, 12).forEach((step, i) => {
        const p = C.parseMove(step.move);
        if (!p) return;
        const col = v.colors[step.seat];
        if (p.kind === "wall") {
          const r = this._wallBox(p.ax, p.ay, p.h);
          el("rect", Object.assign({}, r, { rx: 4, fill: col, opacity: 0.8, stroke: "#fff", "stroke-width": 1.5, "stroke-dasharray": "4 3" }), g);
          const cx = r.x + r.width / 2, cy = r.y + r.height / 2;
          el("circle", { cx, cy, r: 9, fill: col, stroke: "#fff", "stroke-width": 2 }, g);
          this._text(g, cx, cy + 4, i + 1, 11, 750, "#fff");
        } else {
          const [cx, cy] = this._center(p.x, p.y);
          el("circle", { cx, cy, r: CELL * 0.3, fill: col, opacity: 0.88, stroke: "#fff", "stroke-width": 3 }, g);
          this._text(g, cx, cy + 5, i + 1, 14, 750, "#fff");
        }
      });
    }

    _text(g, x, y, s, size, weight, fill) {
      const t = el("text", { x, y, "text-anchor": "middle", "font-size": size, "font-weight": weight, fill, "pointer-events": "none" }, g);
      t.textContent = s;
      return t;
    }

    _drawHover() {
      const g = this.previewLayer;
      if (!g) return;
      while (g.firstChild) g.removeChild(g.firstChild);
      const v = this.view, h = this.hover;
      if (!h || !v || v.inactive) return;
      const col = v.colors[v.dotSeat];
      if (h.kind === "wall") {
        el("rect", Object.assign(this._wallBox(h.ax, h.ay, h.h), { rx: 4, fill: h.legal ? col : "var(--bad)", opacity: h.legal ? 0.6 : 0.5,
          stroke: h.legal ? "#fff" : "var(--bad)", "stroke-width": 1.5 }), g);
      } else if (h.legal) {
        const [cx, cy] = this._center(h.x, h.y);
        el("circle", { cx, cy, r: CELL * 0.34, fill: col, stroke: "#fff", "stroke-width": 3, opacity: 0.5 }, g);
      }
    }
  }

  const api = { Q4Board, CELL, PAD, fmtVisits, CAND_TOP, CAND_OFF, candAlpha };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.Q4Board = Q4Board;
  root.Q4BoardUtil = api;
})(typeof window !== "undefined" ? window : globalThis);
