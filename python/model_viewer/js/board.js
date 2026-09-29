"use strict";
// ---- Quoridor board (board coordinates: cell (c, r), r = 0 is row "1" at the top = Black's goal) ------------
// Wall anchors follow cpp/game/board.h: V (c, r) blocks the east edges of (c, r), (c, r+1); H (c, r) blocks the
// south edges of (c, r), (c+1, r). Canonical (side-to-move) tensors are mapped with canonToBoard below.
const QB = { N: 9, CELL: 40, GAP: 10, PAD: 24 };
QB.PITCH = QB.CELL + QB.GAP;
QB.SIZE = QB.N * QB.CELL + (QB.N - 1) * QB.GAP + 2 * QB.PAD;

const colName = c => String.fromCharCode(97 + c);
function parseMove(s) {
  const kind = s.endsWith("h") ? "h" : s.endsWith("v") ? "v" : "p";
  return { kind, c: s.charCodeAt(0) - 97, r: parseInt(kind === "p" ? s.slice(1) : s.slice(1, -1)) - 1 };
}
// canonical index (rCanon * 9 + c) -> board index for the side to move; wall planes flip only rows 0..7
function canonToBoard(k, pla, wall) {
  const rc = Math.floor(k / 9), c = k % 9;
  if (pla === "B") return k;
  if (wall) return rc < 8 ? (7 - rc) * 9 + c : k;
  return (8 - rc) * 9 + c;
}

function drawQBoard(svg, st, o = {}) {
  const { CELL, GAP, PAD, PITCH, SIZE, N } = QB;
  svg.setAttribute("viewBox", `0 0 ${SIZE} ${SIZE}`);
  svg.innerHTML = "";
  const px = c => PAD + c * PITCH, py = r => PAD + r * PITCH;
  sv("rect", { x: 2, y: 2, width: SIZE - 4, height: SIZE - 4, rx: 16, fill: "var(--board)", stroke: "var(--board-edge)", "stroke-width": 3 }, svg);
  if (!o.small) for (let i = 0; i < N; i++) {
    svText(svg, px(i) + CELL / 2, PAD - 8, colName(i), { "text-anchor": "middle", "font-size": 11, fill: "var(--label)", "font-weight": 600 });
    svText(svg, PAD - 11, py(i) + CELL / 2 + 4, i + 1, { "text-anchor": "middle", "font-size": 11, fill: "var(--label)", "font-weight": 600 });
  }
  const heat = o.cellHeat;
  let hmax = 0;
  if (heat) for (const v of heat.vals) if (v != null) hmax = Math.max(hmax, Math.abs(v));
  if (heat && heat.max) hmax = heat.max;
  for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
    const fill = r === 0 ? "var(--cell-goal-b)" : r === N - 1 ? "var(--cell-goal-w)" : "var(--cell)";
    const cellRect = sv("rect", { x: px(c), y: py(r), width: CELL, height: CELL, rx: 7, fill }, svg);
    const v = heat ? heat.vals[r * 9 + c] : null;
    if (v != null && hmax > 0 && Math.abs(v) > 1e-9) {
      const t = v / hmax;
      const col = heat.kind === "div" ? divColor(t) : heat.kind === "policy" ? null : seqColor(Math.abs(t));
      if (heat.kind === "policy") {
        sv("rect", { x: px(c) + 2, y: py(r) + 2, width: CELL - 4, height: CELL - 4, rx: 6, fill: "var(--s3)", opacity: Math.min(0.92, 0.12 + 0.8 * Math.sqrt(Math.abs(t))) }, svg);
      } else {
        sv("rect", { x: px(c) + 1, y: py(r) + 1, width: CELL - 2, height: CELL - 2, rx: 6, fill: rgbStr(col) }, svg);
      }
    }
    if (o.onCellHover) {
      const hit = sv("rect", { x: px(c) - GAP / 2, y: py(r) - GAP / 2, width: PITCH, height: PITCH, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", ev => o.onCellHover(c, r, ev));
      hit.addEventListener("mouseleave", ev => o.onCellLeave && o.onCellLeave(ev));
      hit.addEventListener("click", ev => o.onCellClick && o.onCellClick(c, r, ev));
      hit.style.cursor = o.onCellClick ? "pointer" : "default";
    }
  }
  // placed walls
  const wallRect = (kind, c, r, attrs) => {
    if (kind === "h") return { x: px(c) - 1, y: py(r) + CELL + 1, width: 2 * CELL + GAP + 2, height: GAP - 2, ...attrs };
    return { x: px(c) + CELL + 1, y: py(r) - 1, width: GAP - 2, height: 2 * CELL + GAP + 2, ...attrs };
  };
  if (st) {
    for (const [c, r] of st.v) sv("rect", wallRect("v", c, r, { rx: 3, fill: "#e8c77d", stroke: "rgba(0,0,0,.35)", "stroke-width": 1 }), svg);
    for (const [c, r] of st.h) sv("rect", wallRect("h", c, r, { rx: 3, fill: "#e8c77d", stroke: "rgba(0,0,0,.35)", "stroke-width": 1 }), svg);
  }
  // candidate-wall heat (policy / final-wall probability): thin bars at the anchor
  if (o.wallHeat) {
    let wmax = o.wallHeat.max || 0;
    if (!wmax) for (const arr of [o.wallHeat.v, o.wallHeat.h]) for (const v of arr || []) if (v != null) wmax = Math.max(wmax, v);
    for (const kind of ["v", "h"]) {
      const arr = o.wallHeat[kind];
      if (!arr) continue;
      for (let k = 0; k < 64; k++) {
        const v = arr[k];
        if (v == null || wmax <= 0 || v / wmax < 0.04) continue;
        const c = k % 8, r = Math.floor(k / 8), t = v / wmax;
        const rect = sv("rect", wallRect(kind, c, r, { rx: 3, fill: o.wallHeat.color || "var(--s5)", opacity: 0.18 + 0.72 * Math.sqrt(t),
          stroke: "rgba(255,255,255,.6)", "stroke-width": t > 0.5 ? 1 : 0 }), svg);
        if (o.wallTip) tipOn(rect, () => o.wallTip(kind, c, r, v));
      }
    }
  }
  // pawns
  if (st) for (const p of ["B", "W"]) {
    const [c, r] = st[p];
    const cx = px(c) + CELL / 2, cy = py(r) + CELL / 2;
    if (st.toMove === p && !o.small) sv("circle", { cx, cy, r: CELL * 0.45, fill: "none", stroke: p === "B" ? "var(--b-acc)" : "var(--w-acc)", "stroke-width": 2, opacity: 0.6 }, svg);
    sv("circle", { cx, cy, r: CELL * 0.31, fill: p === "B" ? "var(--b-pawn)" : "var(--w-pawn)", stroke: p === "B" ? "var(--b-acc)" : "var(--w-acc)", "stroke-width": 3.5, "pointer-events": "none" }, svg);
  }
  // labels on top of everything
  if (heat && heat.labels) {
    for (let k = 0; k < 81; k++) {
      const v = heat.vals[k];
      if (v == null || (heat.labelMin != null && Math.abs(v) < heat.labelMin)) continue;
      const c = k % 9, r = Math.floor(k / 9);
      const dark = heat.kind === "policy" ? Math.abs(v) / hmax > 0.35 : Math.abs(v) / hmax > 0.55;
      svText(svg, px(c) + CELL / 2, py(r) + CELL / 2 + 4, heat.labels(v), { "text-anchor": "middle", "font-size": 11, "font-weight": 700,
        fill: dark ? "#fff" : "#1f2933", "pointer-events": "none" });
    }
  }
  if (o.wallLabels) for (const { kind, c, r, text } of o.wallLabels) {
    const R = wallRect(kind, c, r, {});
    const x = R.x + R.width / 2, y = R.y + R.height / 2;
    sv("rect", { x: x - 17, y: y - 8, width: 34, height: 15, rx: 5, fill: "rgba(20,20,20,.78)", "pointer-events": "none" }, svg);
    svText(svg, x, y + 3.5, text, { "text-anchor": "middle", "font-size": 10, "font-weight": 700, fill: "#fff", "pointer-events": "none" });
  }
  if (o.marks) for (const m of o.marks) {
    if (m.kind === "p") {
      sv("rect", { x: px(m.c) - 2, y: py(m.r) - 2, width: CELL + 4, height: CELL + 4, rx: 9, fill: "none", stroke: m.color || "#fff", "stroke-width": 2.5, "stroke-dasharray": m.dash || null, "pointer-events": "none" }, svg);
    } else {
      sv("rect", wallRect(m.kind, m.c, m.r, { rx: 3, fill: "none", stroke: m.color || "#fff", "stroke-width": 2.5, "stroke-dasharray": m.dash || null, "pointer-events": "none" }), svg);
    }
  }
  return { px, py };
}

// Plain 9x9 heat grid (for input planes, canonical orientation, my goal row on top).
function drawPlane(vals, opts = {}) {
  const S = 9, cs = 10, g = 1, W = S * cs + (S + 1) * g;
  const svg = svgRoot(W, W);
  sv("rect", { x: 0, y: 0, width: W, height: W, fill: "var(--board)", rx: 3 }, svg);
  const max = opts.max || Math.max(1e-9, ...vals.map(Math.abs));
  for (let k = 0; k < 81; k++) {
    const c = k % 9, r = Math.floor(k / 9), v = vals[k];
    const col = opts.div ? rgbStr(divColor(v / max)) : v === 0 ? "var(--cell)" : rgbStr(seqColor(Math.abs(v) / max));
    sv("rect", { x: g + c * (cs + g), y: g + r * (cs + g), width: cs, height: cs, rx: 1.5, fill: col }, svg);
  }
  return svg;
}
