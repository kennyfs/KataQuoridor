/* The 11 x 11 Q4 board as SVG: pawns, walls (in the colour of the seat that placed them), legal pawn destinations,
 * the wall preview under the pointer (legal in the seat colour, illegal red and not clickable), the candidate moves of
 * the analysis, and the principal variation of a hovered candidate. Rotation is set by the app (see core.js).
 * Depends on Q4Core (core.js) only; no state of its own beyond the last view it was given. */
(function (root) {
  const C = root.Q4Core || require("./core.js");
  const CELL = 40, PAD = 26, N = C.N;
  const SVGNS = "http://www.w3.org/2000/svg";
  const BEST = "#56b4e9", OTHER = "#009e73";

  function el(name, attrs, parent) {
    const e = document.createElementNS(SVGNS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  class Q4Board {
    constructor(svg, handlers) {
      this.svg = svg;
      this.h = handlers || {};
      this.rot = 0;
      this.view = null;
      this.hover = null;   // {kind:"pawn"|"wall", ...} under the pointer
      this.svg.setAttribute("viewBox", `0 0 ${N * CELL + 2 * PAD} ${N * CELL + 2 * PAD}`);
      this.svg.addEventListener("mousemove", (e) => this._move(e));
      this.svg.addEventListener("mouseleave", () => { this.hover = null; this._drawHover(); });
      this.svg.addEventListener("click", (e) => this._click(e));
      this.layers = {};
      for (const name of ["cells", "labels", "legal", "walls", "candidates", "pv", "last", "pawns", "hover"]) {
        this.layers[name] = el("g", { "data-layer": name }, this.svg);
      }
    }

    setRotation(rot) { this.rot = ((rot % 4) + 4) % 4; if (this.view) this.render(this.view); }

    // The screen position of a base point (continuous coordinates) in the current rotation.
    _pt(bx, by) { let p = [bx, by]; for (let k = 0; k < this.rot; k++) p = C.rotatePoint(p[0], p[1]); return [PAD + p[0] * CELL, PAD + (N - p[1]) * CELL]; }
    _center(x, y) { return this._pt(x + 0.5, y + 0.5); }

    // Screen -> base point in continuous coordinates (before rotation).
    _base(e) {
      const r = this.svg.getBoundingClientRect();
      const vb = this.svg.viewBox.baseVal;
      const sx = (e.clientX - r.left) * (vb.width / r.width), sy = (e.clientY - r.top) * (vb.height / r.height);
      const vx = (sx - PAD) / CELL, vy = N - (sy - PAD) / CELL;
      return C.viewToBase(vx, vy, this.rot);
    }

    // What the pointer is on: a wall groove, a cell, or nothing (computed from the event itself).
    _pick(e) {
      const b = this._base(e);
      // A pawn destination (a dot, or the cell of a legal pawn move) wins over a groove next to it.
      const cx = Math.floor(b[0]), cy = Math.floor(b[1]);
      if (cx >= 0 && cx < N && cy >= 0 && cy < N) {
        const name = C.cellName(cx, cy);
        const near = Math.hypot(b[0] - (cx + 0.5), b[1] - (cy + 0.5)) < 0.4;
        if (this.view && this.view.legalPawns.has(name) && near) return { kind: "cell", name, x: cx, y: cy };
      }
      if (b[0] >= 0 && b[0] <= N && b[1] >= 0 && b[1] <= N) {
        const g = C.grooveAt(b[0], b[1]);
        if (g) {
          const name = C.wallName(g.ax, g.ay, g.h);
          return { kind: "wall", name, ax: g.ax, ay: g.ay, h: g.h, legal: this.view.legalWalls.has(name) };
        }
      }
      const x = Math.floor(b[0]), y = Math.floor(b[1]);
      if (x >= 0 && x < N && y >= 0 && y < N) return { kind: "cell", name: C.cellName(x, y), x, y };
      return null;
    }

    _move(e) {
      if (!this.view) return;
      const next = this._pick(e);
      const same = JSON.stringify(next) === JSON.stringify(this.hover);
      this.hover = next;
      if (!same) this._drawHover();
      if (this.h.onHover) this.h.onHover(next);
    }

    _click(e) {
      if (!this.view) return;
      const hv = this._pick(e);
      if (!hv) return;
      if (hv.kind === "wall") {
        if (hv.legal && this.h.onMove) this.h.onMove(hv.name);
        else if (this.h.onIllegal) this.h.onIllegal(hv.name);
        return;
      }
      if (this.view.legalPawns.has(hv.name) && this.h.onMove) return this.h.onMove(hv.name);
      if (this.h.onCell) this.h.onCell(hv.name);
    }

    render(view) {
      this.view = view;
      for (const k in this.layers) while (this.layers[k].firstChild) this.layers[k].removeChild(this.layers[k].firstChild);
      this._cells();
      this._walls();
      this._last();
      this._legal();
      this._candidates();
      this._pawns();
      this._pv();
      this._drawHover();
    }

    _cells() {
      const L = this.layers.cells, labels = this.layers.labels;
      for (let x = 0; x < N; x++) for (let y = 0; y < N; y++) {
        const [cx, cy] = this._center(x, y);
        const isCentre = x === C.CENTER[0] && y === C.CENTER[1];
        el("rect", { x: cx - CELL / 2 + 1, y: cy - CELL / 2 + 1, width: CELL - 2, height: CELL - 2, rx: 4,
          fill: isCentre ? "#f6e7a8" : "#f4f1ea", stroke: "#d8d2c4", "stroke-width": 1 }, L);
        if (isCentre) el("circle", { cx, cy, r: 4, fill: "none", stroke: "#b8860b", "stroke-width": 1.5 }, L);
        const t = el("text", { x: cx - CELL / 2 + 3, y: cy + CELL / 2 - 3, "font-size": 8, fill: "#a9b4bf" }, labels);
        t.textContent = C.cellName(x, y);
      }
    }

    _walls() {
      const pos = this.view.position;
      if (!pos) return;
      const seatsOf = this.view.wallSeats || {};
      for (const w of pos.walls || []) {
        const ax = w.charCodeAt(0) - 97, ay = parseInt(w.slice(1, -1), 10) - 1, h = w.endsWith("h");
        const seat = seatsOf[w];
        const dead = seat !== undefined && !pos.alive[seat];
        this._wallRect(ax, ay, h, seat === undefined ? "#555" : this.view.colors[seat], dead ? 0.22 : 1, this.layers.walls);
      }
    }

    _wallRect(ax, ay, h, fill, opacity, parent, extra) {
      const r = C.wallRect(ax, ay, h, 0.14);
      const pts = [[r.x, r.y], [r.x + r.w, r.y], [r.x + r.w, r.y + r.h], [r.x, r.y + r.h]].map(([x, y]) => this._pt(x, y));
      const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
      return el("rect", Object.assign({ x: Math.min(...xs), y: Math.min(...ys), width: Math.max(...xs) - Math.min(...xs),
        height: Math.max(...ys) - Math.min(...ys), rx: 3, fill, "fill-opacity": opacity, stroke: "#222",
        "stroke-width": 0.8, "stroke-opacity": opacity }, extra || {}), parent);
    }

    _last() {
      const m = this.view.lastMove;
      if (!m) return;
      if (m.kind === "pawn") {
        const [cx, cy] = this._center(m.x, m.y);
        el("circle", { cx, cy, r: CELL * 0.42, fill: "none", stroke: "#111", "stroke-width": 2, "stroke-dasharray": "3 2" }, this.layers.last);
      } else {
        this._wallRect(m.ax, m.ay, m.h, "none", 1, this.layers.last, { stroke: "#111", "stroke-width": 2.2, fill: "none" });
      }
    }

    _legal() {
      const v = this.view;
      if (!v.legalDots) return;
      for (const name of v.legalDots) {
        const p = C.parseMove(name);
        if (!p || p.kind !== "pawn") continue;
        const [cx, cy] = this._center(p.x, p.y);
        el("circle", { cx, cy, r: 6, fill: v.colors[v.dotSeat], "fill-opacity": 0.55, stroke: "#fff", "stroke-width": 1 }, this.layers.legal);
      }
    }

    _candidates() {
      const v = this.view;
      if (!v.candidates || !v.candidates.length) return;
      const top = Math.max(...v.candidates.map(c => c.visits)) || 1;
      const L = this.layers.candidates;
      v.candidates.forEach((c, i) => {
        const p = C.parseMove(c.move);
        if (!p) return;
        const col = i === 0 ? BEST : OTHER;
        const op = i === 0 ? 0.85 : 0.25 + 0.5 * (c.visits / top);
        const pctText = C.pct(c.win, 0);
        if (p.kind === "pawn") {
          const [cx, cy] = this._center(p.x, p.y);
          el("circle", { cx, cy, r: CELL * 0.36, fill: col, "fill-opacity": op, stroke: "#fff", "stroke-width": 1 }, L);
          this._label(cx, cy + 3, pctText, 9, "#fff", L);
        } else {
          this._wallRect(p.ax, p.ay, p.h, col, op, L, { "stroke": "#fff" });
          const [mx, my] = this._pt(p.ax + 1, p.ay + 1);
          this._label(mx, my + 3, pctText, 8, "#fff", L);
        }
      });
    }

    _label(x, y, text, size, fill, parent) {
      const t = el("text", { x, y, "text-anchor": "middle", "font-size": size, fill, "font-weight": 600,
        "pointer-events": "none" }, parent);
      t.textContent = text;
      return t;
    }

    _pawns() {
      const v = this.view, pos = v.position;
      if (!pos) return;
      for (let s = 0; s < 4; s++) {
        if (!pos.alive[s] || !pos.pawns[s]) continue;
        const p = C.parseMove(pos.pawns[s]);
        const [cx, cy] = this._center(p.x, p.y);
        const g = el("g", { "data-seat": s }, this.layers.pawns);
        el("circle", { cx, cy, r: CELL * 0.36, fill: v.colors[s], stroke: s === pos.toMove ? "#111" : "#fff",
          "stroke-width": s === pos.toMove ? 2.5 : 1.5 }, g);
        this._label(cx, cy + 4, C.SEAT_LETTERS[s], 11, "#fff", g);
      }
    }

    _pv() {
      const pv = this.view.pv;
      if (!pv || !pv.length) return;
      pv.forEach((step, i) => {
        const p = C.parseMove(step.move);
        if (!p) return;
        const col = this.view.colors[step.seat];
        if (p.kind === "pawn") {
          const [cx, cy] = this._center(p.x, p.y);
          el("circle", { cx, cy, r: 9, fill: "#fff", stroke: col, "stroke-width": 2 }, this.layers.pv);
          this._label(cx, cy + 3.5, String(i + 1), 9, col, this.layers.pv);
        } else {
          const [mx, my] = this._pt(p.ax + 1, p.ay + 1);
          el("circle", { cx: mx, cy: my, r: 9, fill: "#fff", stroke: col, "stroke-width": 2 }, this.layers.pv);
          this._label(mx, my + 3.5, String(i + 1), 9, col, this.layers.pv);
        }
      });
    }

    _drawHover() {
      const L = this.layers.hover;
      while (L.firstChild) L.removeChild(L.firstChild);
      const v = this.view, hv = this.hover;
      if (!v || !hv || v.inactive) return;
      if (hv.kind === "wall") {
        const seat = v.dotSeat;
        const fill = hv.legal ? v.colors[seat] : "#d55e00";
        this._wallRect(hv.ax, hv.ay, hv.h, fill, hv.legal ? 0.55 : 0.8, L, { "stroke": hv.legal ? "#222" : "#7a1f00",
          "stroke-dasharray": hv.legal ? "" : "3 2" });
        const [mx, my] = this._pt(hv.ax + 1, hv.ay + 1);
        this._label(mx, my + 3, hv.legal ? hv.name : "illegal", 8, "#111", L);
      } else if (v.legalPawns.has(hv.name)) {
        const [cx, cy] = this._center(hv.x, hv.y);
        el("circle", { cx, cy, r: CELL * 0.4, fill: "none", stroke: v.colors[v.dotSeat], "stroke-width": 2 }, L);
      }
    }
  }

  const api = { Q4Board, CELL, PAD };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.Q4Board = Q4Board;
})(typeof window !== "undefined" ? window : globalThis);
