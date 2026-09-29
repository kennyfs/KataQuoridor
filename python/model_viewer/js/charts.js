"use strict";
// ---- Small SVG chart helpers (no dependencies). Thin marks, hairline grid, hover tooltips. ------------------

// Horizontal bar chart. items: [{label, value, color, tip, pill}]; values >= 0 (negatives drawn to the left of 0).
function hbarChart(items, o = {}) {
  const W = o.width || 640, rowH = o.rowH || 20, labW = o.labelWidth || 230, padR = 64, top = 6;
  const Hh = top + items.length * rowH + 22;
  const svg = svgRoot(W, Hh);
  const vals = items.map(d => d.value ?? 0);
  const vmax = Math.max(1e-12, ...vals), vmin = Math.min(0, ...vals);
  const x0 = labW + 8, x1 = W - padR;
  const X = v => x0 + (v - vmin) / (vmax - vmin) * (x1 - x0);
  // grid
  const ticks = niceTicks(vmin, vmax, 4);
  for (const t of ticks) {
    sv("line", { x1: X(t), x2: X(t), y1: top - 2, y2: Hh - 18, stroke: "var(--grid)", "stroke-width": 1 }, svg);
    svText(svg, X(t), Hh - 5, (o.tickFmt || fmt.num)(t), { "text-anchor": "middle", "font-size": 10.5, fill: "var(--muted)" });
  }
  sv("line", { x1: X(0), x2: X(0), y1: top - 2, y2: Hh - 18, stroke: "var(--axis)", "stroke-width": 1 }, svg);
  items.forEach((d, i) => {
    const y = top + i * rowH, v = d.value ?? 0;
    const g = sv("g", {}, svg);
    sv("rect", { x: 0, y, width: W, height: rowH, fill: "transparent" }, g);
    if (o.highlight && o.highlight(d)) sv("rect", { x: 0, y, width: W, height: rowH, fill: "var(--sel)", rx: 4 }, g);
    const lab = d.label.length > 40 ? d.label.slice(0, 39) + "…" : d.label;
    svText(g, labW, y + rowH / 2 + 4, lab, { "text-anchor": "end", "font-size": 12, fill: d.dim ? "var(--muted)" : "var(--ink)" });
    const xa = X(Math.min(0, v)), xb = X(Math.max(0, v));
    sv("rect", { x: xa, y: y + 4, width: Math.max(1.5, xb - xa), height: rowH - 8, rx: 3, fill: d.color || "var(--s1)", opacity: d.dim ? 0.45 : 1 }, g);
    svText(g, xb + 5, y + rowH / 2 + 4, (o.valFmt || fmt.num)(v), { "font-size": 11, fill: "var(--ink2)" });
    if (d.tip) tipOn(g, d.tip);
  });
  return svg;
}

function niceTicks(lo, hi, n) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-12; t += step) out.push(+t.toPrecision(12));
  return out;
}

// Heatmap table: rows x cols with a sequential (or diverging) color scale. m[r][c] may be null.
function heatTable(rowLabels, colLabels, m, o = {}) {
  const cw = o.cellW || 92, ch = o.cellH || 22, labW = o.labelWidth || 230, top = 30;
  const W = labW + colLabels.length * cw + 4, Hh = top + rowLabels.length * ch + 4;
  const svg = svgRoot(W, Hh);
  let vmax = 0;
  for (const row of m) for (const v of row) if (v != null) vmax = Math.max(vmax, Math.abs(v));
  if (o.max) vmax = o.max;
  colLabels.forEach((c, j) => svText(svg, labW + j * cw + cw / 2, 18, c, { "text-anchor": "middle", "font-size": 11.5, fill: "var(--ink2)", "font-weight": 600 }));
  rowLabels.forEach((r, i) => {
    const y = top + i * ch;
    svText(svg, labW - 8, y + ch / 2 + 4, r.length > 38 ? r.slice(0, 37) + "…" : r, { "text-anchor": "end", "font-size": 12, fill: "var(--ink)" });
    m[i].forEach((v, j) => {
      const x = labW + j * cw;
      const t = vmax > 0 && v != null ? v / vmax : 0;
      const col = v == null ? "var(--line2)" : rgbStr(o.div ? divColor(t) : seqColor(Math.max(0, t)));
      const cell = sv("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3, fill: col }, svg);
      if (v != null) svText(svg, x + cw / 2, y + ch / 2 + 4, (o.valFmt || fmt.num)(v), { "text-anchor": "middle", "font-size": 10.5,
        fill: Math.abs(t) > 0.55 ? "#fff" : "var(--ink)", "pointer-events": "none" });
      if (o.tip) tipOn(cell, () => o.tip(i, j, v));
    });
  });
  return svg;
}

// Vertical histogram. counts[i] over [edges[i], edges[i+1]).
function histChart(counts, edges, o = {}) {
  const W = o.width || 420, Hh = o.height || 130, padL = 34, padB = 20, top = 6;
  const svg = svgRoot(W, Hh);
  const cmax = Math.max(1, ...counts), n = counts.length, bw = (W - padL - 4) / n;
  const Y = v => top + (1 - v / cmax) * (Hh - top - padB);
  for (const t of niceTicks(0, cmax, 3)) {
    sv("line", { x1: padL, x2: W, y1: Y(t), y2: Y(t), stroke: "var(--grid)" }, svg);
    svText(svg, padL - 5, Y(t) + 3.5, fmt.si(t), { "text-anchor": "end", "font-size": 10, fill: "var(--muted)" });
  }
  counts.forEach((c, i) => {
    const x = padL + i * bw, y = Y(c);
    const g = sv("g", {}, svg);
    sv("rect", { x, y: top, width: bw, height: Hh - top - padB, fill: "transparent" }, g);
    if (c > 0) sv("rect", { x: x + 1, y, width: Math.max(1, bw - 2), height: Hh - padB - y, rx: Math.min(3, bw / 3), fill: o.color || "var(--s1)" }, g);
    tipOn(g, () => `<b>${o.binFmt ? o.binFmt(edges[i], edges[i + 1]) : fmt.num(edges[i]) + " … " + fmt.num(edges[i + 1])}</b><br>${fmt.int(c)}${o.unit || ""}`);
  });
  sv("line", { x1: padL, x2: W, y1: Hh - padB, y2: Hh - padB, stroke: "var(--axis)" }, svg);
  const labIdx = o.labelIdx || [0, Math.floor(n / 2), n];
  for (const i of labIdx) svText(svg, padL + i * bw, Hh - 6, (o.edgeFmt || fmt.num)(edges[i]), { "text-anchor": i === 0 ? "start" : i === n ? "end" : "middle", "font-size": 10, fill: "var(--muted)" });
  return svg;
}

// Line chart. series: [{name, color, pts: [[x, y], ...], dash}]; one y axis.
function lineChart(series, o = {}) {
  const W = o.width || 420, Hh = o.height || 170, padL = 44, padB = 24, top = 10, padR = 10;
  const svg = svgRoot(W, Hh);
  const xs = series.flatMap(s => s.pts.map(p => p[0])), ys = series.flatMap(s => s.pts.map(p => p[1]));
  const xlo = o.xmin ?? Math.min(...xs), xhi = o.xmax ?? Math.max(...xs);
  let ylo = o.ymin ?? Math.min(0, ...ys), yhi = o.ymax ?? Math.max(...ys);
  if (o.logY) { ylo = Math.log10(Math.max(1e-6, Math.min(...ys.filter(v => v > 0)))); yhi = Math.log10(Math.max(...ys)); }
  const X = x => padL + (x - xlo) / Math.max(1e-12, xhi - xlo) * (W - padL - padR);
  const Y = y => { const v = o.logY ? Math.log10(Math.max(1e-6, y)) : y; return top + (1 - (v - ylo) / Math.max(1e-12, yhi - ylo)) * (Hh - top - padB); };
  const yt = o.logY ? niceTicks(Math.floor(ylo), Math.ceil(yhi), 4).map(v => Math.pow(10, v)) : niceTicks(ylo, yhi, 4);
  for (const t of yt) {
    if (o.logY && (Math.log10(t) < ylo - 1e-9 || Math.log10(t) > yhi + 1e-9)) continue;
    sv("line", { x1: padL, x2: W - padR, y1: Y(t), y2: Y(t), stroke: "var(--grid)" }, svg);
    svText(svg, padL - 5, Y(t) + 3.5, (o.yFmt || fmt.num)(t), { "text-anchor": "end", "font-size": 10, fill: "var(--muted)" });
  }
  for (const t of (o.xTicks || niceTicks(xlo, xhi, 5))) svText(svg, X(t), Hh - 8, (o.xFmt || fmt.num)(t), { "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" });
  sv("line", { x1: padL, x2: W - padR, y1: Hh - padB, y2: Hh - padB, stroke: "var(--axis)" }, svg);
  if (o.xLabel) svText(svg, W - padR, Hh - padB - 5, o.xLabel, { "text-anchor": "end", "font-size": 10.5, fill: "var(--muted)" });
  if (o.diag) sv("line", { x1: X(xlo), y1: Y(xlo), x2: X(xhi), y2: Y(xhi), stroke: "var(--axis)", "stroke-width": 1 }, svg);
  for (const s of series) {
    if (s.band) {
      const d = s.band.map(([x, lo], i) => (i ? "L" : "M") + X(x) + " " + Y(lo)).join("") +
        s.band.slice().reverse().map(([x, , hi]) => "L" + X(x) + " " + Y(hi)).join("") + "Z";
      sv("path", { d, fill: s.color, opacity: 0.14 }, svg);
    }
    const d = s.pts.map(([x, y], i) => (i ? "L" : "M") + X(x) + " " + Y(y)).join("");
    if (!s.dotsOnly) sv("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dash || null }, svg);
    if (s.dots) for (const [x, y, r] of s.pts) sv("circle", { cx: X(x), cy: Y(y), r: r || 4, fill: s.color, stroke: "var(--panel)", "stroke-width": 2 }, svg);
  }
  // crosshair tooltip
  const hit = sv("rect", { x: padL, y: top, width: W - padL - padR, height: Hh - top - padB, fill: "transparent" }, svg);
  const cross = sv("line", { y1: top, y2: Hh - padB, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, svg);
  hit.addEventListener("mousemove", ev => {
    const r = svg.getBoundingClientRect(), mx = (ev.clientX - r.left) / r.width * W;
    const xv = xlo + (mx - padL) / (W - padL - padR) * (xhi - xlo);
    let best = null;
    for (const s of series) for (const p of s.pts) if (!best || Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
    if (!best) return;
    cross.setAttribute("x1", X(best[0])); cross.setAttribute("x2", X(best[0])); cross.setAttribute("visibility", "visible");
    const lines = series.map(s => { const p = s.pts.find(q => q[0] === best[0]); return p ? `<span class="sw" style="background:${s.color}"></span>${esc(s.name)}: <b>${(o.yFmt || fmt.num)(p[1])}</b>` : null; }).filter(Boolean);
    showTip(ev, `<b>${o.xName ? o.xName + " " : ""}${(o.xFmt || fmt.num)(best[0])}</b>${o.extraTip ? o.extraTip(best[0]) : ""}<br>` + lines.join("<br>"));
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  return svg;
}

// Grouped vertical bars: cats = [labels], series = [{name, color, vals}]
function groupedBars(cats, series, o = {}) {
  const W = o.width || 460, Hh = o.height || 180, padL = 44, padB = 34, top = 10;
  const svg = svgRoot(W, Hh);
  const all = series.flatMap(s => s.vals).filter(v => v != null);
  const vmax = o.max || Math.max(1e-12, ...all);
  const Y = v => top + (1 - v / vmax) * (Hh - top - padB);
  for (const t of niceTicks(0, vmax, 4)) {
    sv("line", { x1: padL, x2: W, y1: Y(t), y2: Y(t), stroke: "var(--grid)" }, svg);
    svText(svg, padL - 5, Y(t) + 3.5, (o.yFmt || fmt.num)(t), { "text-anchor": "end", "font-size": 10, fill: "var(--muted)" });
  }
  const gw = (W - padL) / cats.length, bw = Math.min(26, (gw - 14) / series.length);
  cats.forEach((c, i) => {
    const gx = padL + i * gw + (gw - bw * series.length) / 2;
    series.forEach((s, j) => {
      const v = s.vals[i];
      if (v == null) return;
      const g = sv("g", {}, svg);
      sv("rect", { x: gx + j * bw, y: top, width: bw, height: Hh - top - padB, fill: "transparent" }, g);
      sv("rect", { x: gx + j * bw + 1, y: Y(v), width: bw - 2, height: Math.max(1, Hh - padB - Y(v)), rx: 3, fill: s.color }, g);
      tipOn(g, `<b>${esc(c)}</b><br><span class="sw" style="background:${s.color}"></span>${esc(s.name)}: <b>${(o.yFmt || fmt.num)(v)}</b>`);
    });
    svText(svg, padL + i * gw + gw / 2, Hh - padB + 15, c, { "text-anchor": "middle", "font-size": 11, fill: "var(--ink2)" });
    if (o.sub) svText(svg, padL + i * gw + gw / 2, Hh - padB + 28, o.sub[i], { "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" });
  });
  sv("line", { x1: padL, x2: W, y1: Hh - padB, y2: Hh - padB, stroke: "var(--axis)" }, svg);
  return svg;
}

function legend(items) {
  return h("div", { class: "legendRow" }, items.map(([name, color]) => h("span", { style: `--c:${color}` }, name)));
}

// Matrix heatmap on a canvas (diverging). vals: Float32Array rows x cols. Returns wrapper element.
function matrixCanvas(vals, rows, cols, o = {}) {
  const maxW = o.maxWidth || 460;
  let sc = Math.max(1, Math.floor(maxW / cols));
  if (o.scale) sc = o.scale;
  const scY = o.scaleY || sc;
  const cv = h("canvas", { width: cols * sc, height: rows * scY });
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(cols * sc, rows * scY);
  let m = o.max || 0;
  if (!m) for (let i = 0; i < vals.length; i++) m = Math.max(m, Math.abs(vals[i]));
  m = m || 1;
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const col = divColor(vals[r * cols + c] / m);
    for (let dy = 0; dy < scY; dy++) for (let dx = 0; dx < sc; dx++) {
      const p = ((r * scY + dy) * cols * sc + c * sc + dx) * 4;
      img.data[p] = col[0]; img.data[p + 1] = col[1]; img.data[p + 2] = col[2]; img.data[p + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  if (o.cssWidth) cv.style.width = o.cssWidth + "px";
  if (o.tip) cv.addEventListener("mousemove", ev => {
    const rc = cv.getBoundingClientRect();
    const c = Math.floor((ev.clientX - rc.left) / rc.width * cols), r = Math.floor((ev.clientY - rc.top) / rc.height * rows);
    if (r >= 0 && r < rows && c >= 0 && c < cols) showTip(ev, o.tip(r, c, vals[r * cols + c]));
  });
  cv.addEventListener("mouseleave", hideTip);
  return cv;
}

function divLegend(maxAbs, label) {
  const w = 160, svg = svgRoot(w + 90, 26);
  for (let i = 0; i < 40; i++) sv("rect", { x: 40 + i * w / 40, y: 4, width: w / 40 + 0.5, height: 9, fill: rgbStr(divColor(i / 19.5 - 1)) }, svg);
  svText(svg, 36, 12, fmt.num(-maxAbs, 2), { "text-anchor": "end", "font-size": 10, fill: "var(--muted)" });
  svText(svg, 44 + w, 12, "+" + fmt.num(maxAbs, 2), { "font-size": 10, fill: "var(--muted)" });
  if (label) svText(svg, 40 + w / 2, 24, label, { "text-anchor": "middle", "font-size": 10, fill: "var(--muted)" });
  svg.style.width = "250px";
  return svg;
}
