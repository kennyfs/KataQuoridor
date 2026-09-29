"use strict";
// ---- Tab 1: architecture diagram + detail panel ----------------------------------------------------------------
const KINDS = {
  input: ["Input", "var(--s6)"], conv: ["Conv", "var(--s1)"], linear: ["Linear", "var(--s7)"], attn: ["Attention", "var(--s2)"],
  ffn: ["FFN", "var(--s3)"], pool: ["Pool", "var(--s4)"], output: ["Output", "var(--s5)"], norm: ["Norm/act", "var(--axis)"],
  add: ["Add", "var(--axis)"], block: ["Block", "var(--ink2)"], generic: ["Module", "var(--axis)"],
};
let NODES = new Map();
let selNodeEl = null;

function renderArch() {
  NODES = new Map();
  const m = DATA.meta, cfg = m.config;
  const attnLayers = DATA.attention ? DATA.attention.layers.length : 0;
  const tiles = $("archTiles"); tiles.innerHTML = "";
  tiles.append(
    tile("Parameters", fmt.si(m.numParams), fmt.int(m.numParams) + (m.swa ? " · SWA weights" : "")),
    tile("Trunk", `${cfg.block_kind.length} blocks`, `${cfg.trunk_num_channels} ch · mid ${cfg.mid_num_channels}`),
    tile("Attention", `${attnLayers} layers`, `${cfg.transformer_heads || "-"} heads × ${cfg.attention_query_head_dim || "-"}`),
    tile("Board", `${cfg.pos_len}×${cfg.pos_len}`, `17 spatial + 15 global inputs`),
    tile("Trained on", fmt.si(m.trainState.global_step_samples), "samples · " + fmt.si(m.trainState.total_num_data_rows) + " rows"),
  );
  // parameter budget
  const sum = pre => Object.entries(DATA.params).filter(([n]) => pre.some(p => n.startsWith(p))).reduce((a, [, p]) => a + p.n, 0);
  const parts = [["Stem", sum(["conv_spatial", "linear_global"]), "var(--s7)", "stem"]];
  cfg.block_kind.forEach((_, i) => parts.push([`Block ${i + 1}`, sum([`blocks.${i}.`]), "var(--s1)", `blocks.${i}`]));
  parts.push(["Policy head", sum(["policy_head"]), "var(--s2)", "policy_head"], ["Value head", sum(["value_head"]), "var(--s3)", "value_head"],
    ["Other", m.numParams - 0, "var(--axis)", null]);
  const known = parts.slice(0, -1).reduce((a, p) => a + p[1], 0);
  parts[parts.length - 1][1] = m.numParams - known;
  const bud = $("archBudget"); bud.innerHTML = "";
  const bar = h("div", { class: "bar" }), labs = h("div", { class: "labels" });
  for (const [name, n, col, target] of parts) {
    if (n <= 0) continue;
    const seg = h("div", { style: `flex:${n};background:${col}`, onclick: () => target && scrollToNode(target) });
    tipOn(seg, `<b>${name}</b><br>${fmt.int(n)} params (${fmt.pct(n / m.numParams)})`);
    bar.append(seg);
    labs.append(h("span", { style: `--c:${col}` }, `${name} ${fmt.pct(n / m.numParams, 0)}`));
  }
  bud.append(h("div", { class: "ttl" }, "Parameter budget"), bar, labs);
  // depth profile
  const act = DATA.activations || {};
  const nb = cfg.block_kind.length;
  const trunkPts = [], resPts = [];
  for (let i = 0; i < nb; i++) { const a = act[`blocks.${i}`]; if (a) { trunkPts.push([i + 1, a.in]); resPts.push([i + 1, a.out]); } }
  if (act.norm_trunkfinal) trunkPts.push([nb + 1, act.norm_trunkfinal.in]);
  if (trunkPts.length) {
    const box = h("div", { class: "chart" }, h("div", { class: "ttl" }, "Residual stream on real positions"),
      h("div", { class: "sub" }, "RMS of the trunk entering each block vs. the RMS of what the block adds (x = block; last point = trunk before the final norm)."));
    box.append(lineChart([{ name: "Trunk RMS (in)", color: "var(--s1)", pts: trunkPts, dots: true }, { name: "Block residual RMS", color: "var(--s3)", pts: resPts, dots: true }],
      { width: 560, height: 150, xTicks: trunkPts.map(p => p[0]), xFmt: x => x > nb ? "final" : "B" + x, xName: "" }));
    box.append(legend([["Trunk RMS (in)", "var(--s1)"], ["Block residual RMS", "var(--s3)"]]));
    bud.append(h("div", { style: "margin-top:10px" }, box));
  }
  $("kindLegend").innerHTML = "";
  $("kindLegend").append(...["input", "conv", "linear", "attn", "ffn", "pool", "output", "norm"].map(k => h("span", { style: `--c:${KINDS[k][1]}` }, KINDS[k][0])),
    h("span", { style: "--c:var(--s3)" }, "⊕ residual add"));

  const dg = $("diagram"); dg.innerHTML = "";
  const flow = h("div", { class: "flow" });
  DATA.arch.forEach((n, i) => {
    if (i) flow.append(arrow(n.shape || (n.children && n.children[0] && n.children[0].shape)));
    flow.append(renderNode(n));
  });
  dg.append(flow);
  buildArchNav();
}

function arrow(shape) {
  const a = h("div", { class: "arrow" });
  if (shape) a.append(h("span", { class: "alab" }, fmt.shape(shape)));
  return a;
}

function renderNode(n) {
  if (n.kind === "row") return h("div", { class: "nrow" }, n.children.map(renderNode));
  if (n.kind === "col") {
    const c = h("div", { class: "ncol" });
    n.children.forEach((ch, i) => { if (i) c.append(arrow()); c.append(renderNode(ch)); });
    return c;
  }
  if (n.kind === "trunk") {
    const c = h("div", { class: "flow", id: "n-trunk" });
    n.children.forEach((ch, i) => { if (i) c.append(arrow(ch.shape)); c.append(renderNode(ch)); });
    return c;
  }
  if (n.kind === "heads") return h("div", { class: "heads" }, n.children.map(renderNode));
  if (n.kind === "head") {
    const c = h("div", { class: "head", id: "n-" + n.id }, h("div", { class: "hh" }, n.title, h("span", { class: "st" }, fmt.int(n.nparams) + " params")));
    const f = h("div", { class: "flow" });
    n.children.forEach((ch, i) => { if (i) f.append(arrow(ch.shape)); f.append(renderNode(ch)); });
    c.append(f);
    return c;
  }
  if (n.kind === "block") return renderBlock(n);
  return renderLeaf(n);
}

function renderBlock(n) {
  NODES.set(n.id, n);
  const a = (DATA.activations || {})[n.id];
  const ratio = a && a.in ? a.out / a.in : null;
  const b = h("div", { class: "block", id: "n-" + n.id });
  const hd = h("div", { class: "bh", onclick: ev => { if (ev.target.classList.contains("caret")) b.classList.toggle("collapsed"); else selectNode(n, hd); } },
    h("span", { class: "caret", title: "Collapse" }, "▾"), h("span", { class: "t" }, n.title), h("span", { class: "st" }, n.sub + " · " + fmt.int(n.nparams) + " params"),
    ratio != null ? h("span", { class: "resbar", title: "RMS of the block's residual / RMS of the trunk it is added to" },
      h("i", { style: `width:${Math.min(80, ratio * 80)}px` }), "adds " + fmt.pct(ratio, 0) + " of trunk RMS") : null);
  const body = h("div", { class: "bbody" });
  n.children.forEach((ch, i) => {
    if (i) body.append(arrow(ch.shape));
    const e = renderLeaf(ch);
    if (ch.residual) e.classList.add("subres");
    body.append(e);
  });
  body.append(h("div", { class: "resjoin" }, "⊕ added to the trunk (residual)"));
  b.append(h("div", { class: "resrail" }), hd, body);
  return b;
}

function mainWeight(n) {
  for (const p of n.params || []) { const d = DATA.params[p]; if (d && d.shape.length >= 2 && d.n >= 64 && !p.includes("rope")) return p; }
  return null;
}

function renderLeaf(n) {
  NODES.set(n.id, n);
  const [kname, kc] = KINDS[n.kind] || KINDS.generic;
  const e = h("div", { class: "node", id: "n-" + n.id.replace(/\./g, "_"), style: `--kc:${kc}` },
    h("div", { class: "t" }, n.title, h("span", { class: "kind" }, kname)),
    n.sub ? h("div", { class: "st" }, n.sub) : null);
  const chips = h("div", { class: "chips" });
  if (n.shape) chips.append(h("span", { class: "chip" }, "out " + fmt.shape(n.shape)));
  if (n.nparams) chips.append(h("span", { class: "chip" }, fmt.int(n.nparams) + " params"));
  const a = (DATA.activations || {})[n.id];
  if (n.residual && a && a.in) chips.append(h("span", { class: "chip res", title: "residual RMS / input RMS" }, "⊕ " + fmt.pct(a.out / a.in, 0)));
  e.append(chips);
  const mw = mainWeight(n);
  if (mw) { e.classList.add("hasthumb"); e.append(thumbCanvas(mw)); }
  e.addEventListener("click", ev => { ev.stopPropagation(); selectNode(n, e); });
  return e;
}

function thumbCanvas(pname) {
  const p = DATA.params[pname], v = paramValues(pname);
  const rows = p.shape[0], cols = p.n / rows, W = 68, H = 40;
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const r0 = Math.floor(y * rows / H), r1 = Math.max(r0 + 1, Math.floor((y + 1) * rows / H));
    const c0 = Math.floor(x * cols / W), c1 = Math.max(c0 + 1, Math.floor((x + 1) * cols / W));
    let s = 0, k = 0;
    for (let r = r0; r < r1 && r < rows; r++) for (let c = c0; c < c1 && c < cols; c++) { s += v[r * cols + c]; k++; }
    out[y * W + x] = k ? s / k : 0;
  }
  const cv = matrixCanvas(out, H, W, { scale: 1 });
  cv.classList.add("thumb");
  cv.title = `${pname} ${fmt.shape(p.shape)}`;
  return cv;
}

function scrollToNode(id) {
  const el = document.getElementById("n-" + id) || document.getElementById("n-" + id.replace(/\./g, "_"));
  if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}

function buildArchNav() {
  const nav = $("archNav"); nav.innerHTML = "";
  nav.append(h("div", { class: "ohead" }, "Outline"));
  const links = [["Inputs", "n-in_spatial"], ["Stem", "n-conv_spatial"]];
  DATA.meta.config.block_kind.forEach((_, i) => links.push([`Block ${i + 1}`, `n-blocks.${i}`]));
  links.push(["Trunk final", "n-norm_trunkfinal"], ["Policy head", "n-policy_head"], ["Value head", "n-value_head"]);
  const as = [];
  for (const [lab, id] of links) {
    const a = h("a", { href: "#", onclick: ev => { ev.preventDefault(); const e = document.getElementById(id); if (e) e.scrollIntoView({ behavior: "smooth", block: "start" }); } }, lab);
    a.dataset.target = id; as.push(a); nav.append(a);
  }
  const wrap = $("diagramWrap");
  wrap.onscroll = () => {
    let cur = as[0];
    for (const a of as) { const e = document.getElementById(a.dataset.target); if (e && e.getBoundingClientRect().top < wrap.getBoundingClientRect().top + 120) cur = a; }
    as.forEach(a => a.classList.toggle("on", a === cur));
  };
  wrap.onscroll();
}

// ---- detail panel ------------------------------------------------------------------------------------------------
function selectNode(n, el) {
  if (selNodeEl) selNodeEl.classList.remove("sel");
  selNodeEl = el; if (el) el.classList.add("sel");
  const d = $("detail"); d.innerHTML = ""; d.scrollTop = 0;
  const [kname, kc] = KINDS[n.kind] || KINDS.generic;
  d.append(h("h2", {}, n.title, " ", h("span", { class: "pill", style: `color:${kc}` }, kname)));
  if (n.sub) d.append(h("div", { style: "color:var(--ink2)" }, n.sub));
  if (n.desc) d.append(h("div", { class: "desc" }, n.desc));
  const kv = h("dl", { class: "kv" });
  const add = (k, v) => kv.append(h("dt", {}, k), h("dd", {}, v));
  add("Module", n.id);
  if (n.shape) add("Output", fmt.shape(n.shape));
  add("Parameters", fmt.int(n.nparams));
  const a = (DATA.activations || {})[n.id];
  if (a && a.in) { add("Input RMS", fmt.num(a.in)); add("Residual RMS", fmt.num(a.out) + ` (${fmt.pct(a.out / a.in, 0)} of input)`); }
  d.append(kv);

  if (n.id === "in_spatial") detailSpatialInput(d);
  if (n.id === "in_global") detailGlobalInput(d);
  if (n.id === "conv_spatial") detailStem(d);
  if (n.id === "linear_global") detailLinearGlobal(d);
  if (n.kind === "attn") detailAttention(d, n);
  if (n.kind === "block") detailBlock(d, n);
  if (n.variants) d.append(h("h3", {}, "Policy variants"), h("ol", { start: 0 }, n.variants.map(v => h("li", {}, v))));

  const params = n.kind === "block" ? [] : (n.params || []);
  if (params.length) d.append(h("h3", {}, "Parameters"));
  for (const p of params) d.append(paramCard(p));
}

function paramCard(name) {
  const p = DATA.params[name];
  const c = h("div", { class: "pcard" }, h("div", { class: "pn" }, name, "  ", h("span", { class: "chip" }, fmt.shape(p.shape))),
    h("div", { class: "pstats" }, `n ${fmt.int(p.n)} · mean ${fmt.num(p.mean)} · std ${fmt.num(p.std)} · |max| ${fmt.num(p.absmax)}` + (p.erank ? ` · eff. rank ${fmt.num(p.erank, 3)}` : "")));
  const v = paramValues(name);
  if (!v) return c;
  if (name.endsWith("rope_freqs")) { c.append(ropeChart(v, p.shape)); return c; }
  const sh = p.shape;
  if (sh.length >= 2 && p.n / sh[0] > 1) {
    const rows = sh[0], cols = p.n / rows;
    const cw = Math.max(1, Math.min(10, Math.floor(440 / cols))), rh = Math.max(1, Math.min(10, Math.floor(260 / rows)));
    const cv = matrixCanvas(v, rows, cols, { scale: 1, tip: (r, cc, val) => `<b>[${r}, ${cc}]</b> ${sh.length === 4 && sh[2] > 1 ? `(in ${Math.floor(cc / (sh[2] * sh[3]))}, k ${cc % (sh[2] * sh[3])})` : ""}<br>${fmt.num(val, 4)}` });
    cv.style.width = cols * cw + "px"; cv.style.height = rows * rh + "px";
    c.append(h("div", { class: "canvasWrap" }, cv), h("div", { class: "caption" }, `rows = output ${sh.length === 4 && sh[2] > 1 ? "channels, cols = input × kernel" : "channels, cols = input channels"}`), divLegend(p.absmax));
  } else {
    c.append(vectorBars(v));
  }
  const mini = h("div", { class: "minirow" });
  if (p.hist) {
    const edges = p.hist.map((_, i) => -p.absmax + 2 * p.absmax * i / p.hist.length).concat([p.absmax]);
    mini.append(h("div", {}, h("div", { class: "caption" }, "Histogram"), histChart(p.hist, edges, { width: 260, height: 110, color: "var(--s1)", labelIdx: [0, p.hist.length / 2, p.hist.length] })));
  }
  if (p.sv) mini.append(h("div", {}, h("div", { class: "caption" }, "Singular values"),
    lineChart([{ name: "σ", color: "var(--s7)", pts: p.sv.map((s, i) => [i + 1, s]) }], { width: 260, height: 110, xName: "#" })));
  c.append(mini);
  return c;
}

function vectorBars(v) {
  const n = v.length, W = 440, Hh = 70, bw = W / n;
  const svg = svgRoot(W, Hh);
  let m = 0; for (const x of v) m = Math.max(m, Math.abs(x));
  m = m || 1;
  sv("line", { x1: 0, x2: W, y1: Hh / 2, y2: Hh / 2, stroke: "var(--axis)" }, svg);
  for (let i = 0; i < n; i++) {
    const y = v[i] / m * (Hh / 2 - 3);
    const r = sv("rect", { x: i * bw + 0.3, y: y > 0 ? Hh / 2 - y : Hh / 2, width: Math.max(0.8, bw - 0.6), height: Math.max(0.5, Math.abs(y)), fill: v[i] >= 0 ? "var(--s1)" : "var(--s8)" }, svg);
    tipOn(r, `<b>[${i}]</b> ${fmt.num(v[i], 4)}`);
  }
  return svg;
}

function ropeChart(v, shape) {
  const [H, P] = shape;
  const box = h("div", {}, h("div", { class: "caption" }, "Learned 2D rotary frequencies (ωx, ωy) per head; each dot is one rotated pair. Wavelength in cells = 2π/|ω|."));
  const row = h("div", { class: "minirow" });
  let m = 0; for (const x of v) m = Math.max(m, Math.abs(x));
  for (let hd = 0; hd < H; hd++) {
    const S = 120, svg = svgRoot(S, S), X = x => S / 2 + x / m * (S / 2 - 8), Y = y => S / 2 - y / m * (S / 2 - 8);
    sv("line", { x1: 0, x2: S, y1: S / 2, y2: S / 2, stroke: "var(--grid)" }, svg);
    sv("line", { y1: 0, y2: S, x1: S / 2, x2: S / 2, stroke: "var(--grid)" }, svg);
    for (let p = 0; p < P; p++) {
      const wx = v[(hd * P + p) * 2], wy = v[(hd * P + p) * 2 + 1], wl = 2 * Math.PI / Math.max(1e-6, Math.hypot(wx, wy));
      const c = sv("circle", { cx: X(wx), cy: Y(wy), r: 4, fill: "var(--s2)", stroke: "var(--panel)", "stroke-width": 1.5 }, svg);
      tipOn(c, `<b>Head ${hd + 1}, pair ${p}</b><br>ωx ${fmt.num(wx)} · ωy ${fmt.num(wy)}<br>wavelength ${fmt.num(wl)} cells`);
    }
    svText(svg, 4, 12, "H" + (hd + 1), { "font-size": 10.5, fill: "var(--muted)", "font-weight": 600 });
    svg.style.width = "118px";
    row.append(h("div", { style: "flex:none;min-width:0" }, svg));
  }
  box.append(row);
  return box;
}

function detailBlock(d, n) {
  const act = DATA.activations || {};
  const items = n.children.filter(c => c.residual && act[c.id]).map(c => ({ label: c.title + " · " + c.id.split(".").slice(-1)[0], value: act[c.id].out / act[c.id].in, color: KINDS[c.kind][1] }));
  if (items.length) {
    d.append(h("h3", {}, "Inner residual updates"), h("div", { class: "caption" }, "RMS of each sub-layer's residual relative to the 96-channel stream it is added to."),
      hbarChart(items, { width: 440, labelWidth: 170, valFmt: x => fmt.pct(x, 0), tickFmt: x => fmt.pct(x, 0) }));
  }
  d.append(h("h3", {}, "Contents"));
  for (const c of n.children) d.append(h("div", { style: "font-size:13px;margin:2px 0" }, h("a", { href: "#", onclick: ev => { ev.preventDefault(); scrollToNode(c.id); selectNode(c, document.getElementById("n-" + c.id.replace(/\./g, "_"))); } }, c.title), " — ", c.sub, " · ", fmt.int(c.nparams)));
}

function exampleInputs(e) { return DATA.attention && DATA.attention.examples[e]; }

function detailSpatialInput(d) {
  const f = DATA.features.spatial, st = DATA.importance && DATA.importance.stem;
  if (DATA.attention) {
    let cur = 0;
    const host = h("div", {});
    const draw = () => {
      host.innerHTML = "";
      const ex = exampleInputs(cur), stt = DATA.attention.states[cur];
      const bsvg = svgRoot(10, 10); drawQBoard(bsvg, stt);
      host.append(h("div", { class: "boardBox", style: "max-width:260px;margin:6px 0" }, bsvg),
        h("div", { class: "caption" }, `${stt.toMove === "B" ? "Black" : "White"} to move. The planes below are in the mover's canonical view (mover's goal row on top).`));
      const grid = h("div", { class: "planeGrid" });
      f.forEach((ff, i) => {
        const pl = h("div", { class: "plane", title: ff.desc }, drawPlane(ex.spatial[i], { max: 1 }), `S${i} ${ff.name}`);
        grid.append(pl);
      });
      host.append(grid);
    };
    d.append(h("h3", {}, "Example position"), selectBox(DATA.attention.labels.map((l, i) => [String(i), l]), "0", v => { cur = +v; draw(); }), host);
    draw();
  }
  d.append(h("h3", {}, "Channels"));
  const tb = h("table", { class: "t" }, h("tr", {}, h("th", {}, "Channel"), h("th", {}, "mean"), h("th", {}, "conv |W|")));
  f.forEach((ff, i) => tb.append(h("tr", { title: ff.desc }, h("td", {}, `S${i} ${ff.name}`, ff.continuous ? h("span", { class: "pill", title: "continuous at inference, binarized in training data" }, "⚠ continuous") : null),
    h("td", {}, st ? fmt.num(st.spatial_mean[i]) : "—"), h("td", {}, st ? fmt.num(st.spatial_norm[i]) : "—"))));
  d.append(tb);
}

function detailGlobalInput(d) {
  const f = DATA.features.global, st = DATA.importance && DATA.importance.stem;
  if (DATA.attention) {
    const ex = exampleInputs(0);
    d.append(h("h3", {}, "Values for example 1"));
    d.append(hbarChart(f.map((ff, i) => ({ label: `G${i} ${ff.name}`, value: ex.global[i], color: "var(--s2)" })), { width: 440, labelWidth: 200, rowH: 18 }));
  }
  d.append(h("h3", {}, "Features"));
  const tb = h("table", { class: "t" }, h("tr", {}, h("th", {}, "Feature"), h("th", {}, "mean"), h("th", {}, "|W| col")));
  f.forEach((ff, i) => tb.append(h("tr", { title: ff.desc }, h("td", {}, `G${i} ${ff.name}`), h("td", {}, st ? fmt.num(st.global_mean[i]) : "—"), h("td", {}, st ? fmt.num(st.global_norm[i]) : "—"))));
  d.append(tb);
}

function detailStem(d) {
  const st = DATA.importance && DATA.importance.stem;
  if (!st) return;
  d.append(h("h3", {}, "Kernel footprint per input plane"),
    h("div", { class: "caption" }, "L2 norm over the output channels of each 3×3 kernel tap: where around a cell the net reads each plane (center = the cell itself, top = toward the mover's goal)."));
  const grid = h("div", { class: "planeGrid" });
  let m = 0; for (const fp of st.spatial_footprint) for (const x of fp) m = Math.max(m, x);
  DATA.features.spatial.forEach((ff, i) => {
    const S = 3, cs = 22, svg = svgRoot(S * cs, S * cs);
    st.spatial_footprint[i].forEach((v, k) => {
      const r = sv("rect", { x: (k % 3) * cs + 1, y: Math.floor(k / 3) * cs + 1, width: cs - 2, height: cs - 2, rx: 3, fill: rgbStr(seqColor(v / m)) }, svg);
      tipOn(r, `<b>S${i} ${esc(ff.name)}</b><br>tap (${Math.floor(k / 3) - 1}, ${(k % 3) - 1}): ${fmt.num(v)}`);
    });
    grid.append(h("div", { class: "plane" }, svg, `S${i} ${ff.name}`));
  });
  d.append(grid);
}

function detailLinearGlobal(d) {
  const v = paramValues("linear_global.weight"), [C, G] = DATA.params["linear_global.weight"].shape;
  const t = new Float32Array(G * C);
  for (let c = 0; c < C; c++) for (let g = 0; g < G; g++) t[g * C + c] = v[c * G + g];
  const cv = matrixCanvas(t, G, C, { scale: 1, tip: (r, c, val) => `<b>G${r} ${esc(DATA.features.global[r].name)}</b> → ch ${c}<br>${fmt.num(val, 4)}` });
  cv.style.width = "440px"; cv.style.height = G * 12 + "px";
  d.append(h("h3", {}, "Weights by global feature (rows) × trunk channel"), h("div", { class: "canvasWrap" }, cv),
    h("div", { class: "caption" }, "Rows in feature order G0 … G14."));
}

// ---- attention explorer -------------------------------------------------------------------------------------------
const _attnCache = new Map();
function attnArray(e) {
  if (!_attnCache.has(e)) _attnCache.set(e, b64Bytes(DATA.attention.examples[e].q));
  return _attnCache.get(e);
}
function detailAttention(d, n) {
  const A = DATA.attention, L = A ? A.layers.indexOf(n.attn_name) : -1;
  const hs = DATA.attentionHeads;
  if (hs) {
    const li = hs.layers.indexOf(n.attn_name);
    d.append(h("h3", {}, "Head statistics (all layers)"), h("div", { class: "caption" }, "Mean Manhattan distance (in cells) between a query cell and the cells it attends to, averaged over 512 training positions. This layer is highlighted."));
    const rows = hs.layers.map((l, i) => (i === li ? "▶ " : "") + l.replace(/^block/, "B").replace(".blockstack.", " "));
    d.append(heatTable(rows, hs.stats[0].map((_, j) => "H" + (j + 1)), hs.stats.map(r => r.map(s => s.mean_dist)), { labelWidth: 120, cellW: 70, valFmt: x => fmt.fix(x, 2),
      tip: (i, j) => { const s = hs.stats[i][j]; return `<b>${esc(hs.layers[i])} H${j + 1}</b><br>mean dist ${fmt.fix(s.mean_dist, 2)} · entropy ${fmt.fix(s.entropy, 2)} / ${fmt.fix(hs.uniform_entropy, 2)}<br>self ${fmt.pct(s.self)} · my pawn ${fmt.pct(s.to_my_pawn)} · opp pawn ${fmt.pct(s.to_opp_pawn)}<br>my path ${fmt.pct(s.to_my_path)} · opp path ${fmt.pct(s.to_opp_path)}`; } }));
    if (li >= 0) {
      const tb = h("table", { class: "t" }, h("tr", {}, ...["Head", "entropy", "self", "my pawn", "opp pawn", "my path", "opp path"].map(x => h("th", {}, x))));
      hs.stats[li].forEach((s, j) => tb.append(h("tr", {}, h("td", {}, "H" + (j + 1)), h("td", {}, fmt.fix(s.entropy, 2)), h("td", {}, fmt.pct(s.self)), h("td", {}, fmt.pct(s.to_my_pawn)), h("td", {}, fmt.pct(s.to_opp_pawn)), h("td", {}, fmt.pct(s.to_my_path)), h("td", {}, fmt.pct(s.to_opp_path)))));
      d.append(h("div", { class: "caption", style: "margin-top:8px" }, "Attention mass on: the query cell itself, each pawn's cell, and each player's shortest-path cells (uniform would be 1/81 per cell)."), tb);
    }
  }
  if (!A || L < 0) return;
  let ex = 0, query = null, mode = "query";
  const host = h("div", {});
  const [LL, H, S] = A.examples[0].shape;
  const draw = () => {
    host.innerHTML = "";
    const st = A.states[ex], pla = st.toMove, arr = attnArray(ex);
    if (!query) query = st[pla].slice();
    const qBoard = query[1] * 9 + query[0];
    const qCanon = canonToBoard(qBoard, pla, false);   // the mapping is an involution
    const grid = h("div", { style: "display:grid;grid-template-columns:1fr 1fr;gap:8px" });
    for (let hd = 0; hd < H; hd++) {
      const vals = new Array(81).fill(0);
      for (let k = 0; k < S; k++) {
        let p;
        if (mode === "query") { const u = arr[((L * H + hd) * S + qCanon) * S + k] / 255; p = u * u; }
        else { p = 0; for (let q = 0; q < S; q++) { const u = arr[((L * H + hd) * S + q) * S + k] / 255; p += u * u; } p /= S; }
        vals[canonToBoard(k, pla, false)] = p;
      }
      const svg = svgRoot(10, 10);
      drawQBoard(svg, st, { small: true, cellHeat: { vals, kind: "seq" }, marks: mode === "query" ? [{ kind: "p", c: query[0], r: query[1], color: "#f5d58a" }] : [],
        onCellHover: (c, r, ev) => showTip(ev, `H${hd + 1}: ${colName(c)}${r + 1} gets <b>${fmt.pct(vals[r * 9 + c])}</b>`), onCellLeave: hideTip,
        onCellClick: (c, r) => { query = [c, r]; mode = "query"; draw(); } });
      grid.append(h("div", {}, h("div", { class: "caption" }, "Head " + (hd + 1)), svg));
    }
    host.append(grid);
  };
  d.append(h("h3", {}, "Attention maps"),
    h("div", { class: "caption" }, "Click a cell to make it the query; the heat shows where that cell's token looks. Default query: the mover's pawn."),
    h("div", { class: "ctrl" }, selectBox(A.labels.map((l, i) => [String(i), `${i + 1}. ${l}`]), "0", v => { ex = +v; query = null; draw(); }),
      segmented([["query", "From query cell"], ["mean", "Mean received"]], mode, v => { mode = v; draw(); })), host);
  draw();
}
