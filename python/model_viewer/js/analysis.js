"use strict";
// ---- Tab 2: feature importance, input parity, calibration, search -----------------------------------
const METRICS = [
  ["kl", "Policy KL (output change)", "Mean KL(original policy || ablated policy).", x => fmt.fix(x, 3)],
  ["dwin", "|Δ P(win)|", "Mean absolute change of the side-to-move win probability.", x => fmt.pct(x, 1)],
  ["dmargin", "|Δ margin|", "Mean absolute change of the predicted game margin (moves).", x => fmt.fix(x, 2)],
  ["top1", "Top move changed", "Fraction of positions whose top policy move changes.", x => fmt.pct(x, 0)],
  ["dpol", "Δ policy loss", "Increase of the policy cross-entropy against the 600-visit search targets.", x => fmt.sgn(x, 3)],
  ["dval", "Δ value loss", "Increase of the value cross-entropy against the game outcomes.", x => fmt.sgn(x, 4)],
  ["dmar", "Δ margin loss", "Increase of the margin Huber loss against the final margins.", x => fmt.sgn(x, 3)],
];
const GXI = [["gxi_value", "Grad × input: value"], ["gxi_policy", "Grad × input: policy"], ["gxi_margin", "Grad × input: margin"], ["wscaled", "First-layer |W| × input RMS"]];
let featState = { metric: "kl", scope: "single" };

function card(id, title, lead, cls) {
  const c = h("section", { class: "card " + (cls || ""), id }, h("h2", {}, title));
  if (lead) c.append(h("div", { class: "lead", html: lead }));
  return c;
}

function renderAnalysis() {
  const body = $("featBody"); body.innerHTML = "";
  const nav = $("featNav"); nav.innerHTML = ""; nav.append(h("div", { class: "ohead" }, "Sections"));
  const sections = [];
  const add = (c, lab) => { body.append(c); sections.push([lab, c.id]); };
  add(summaryCard(), "Summary");
  if (DATA.inputParity) add(parityCard(), "Input parity");
  add(importanceCard(), "Feature importance");
  add(phaseCard(), "By game phase");
  add(stemCard(), "First-layer weights");
  if (DATA.calibration) add(calibrationCard(), "Calibration");
  if (DATA.mcts && DATA.mcts.summary) { add(searchCard(), "Net vs search"); add(browserCard(), "Position browser"); }
  const as = sections.map(([lab, id]) => { const a = h("a", { href: "#", onclick: ev => { ev.preventDefault(); $(id).scrollIntoView({ behavior: "smooth" }); } }, lab); a.dataset.t = id; nav.append(a); return a; });
  body.onscroll = () => { let cur = as[0]; for (const a of as) if ($(a.dataset.t).getBoundingClientRect().top < body.getBoundingClientRect().top + 100) cur = a; as.forEach(a => a.classList.toggle("on", a === cur)); };
  body.onscroll();
}

function allFeatureRows() {
  const P = DATA.importance.permutation;
  return [
    ...P.spatial.map(e => ({ kind: "s", idx: e.idx, label: featLabel("s", e.idx), e, color: "var(--s1)" })),
    ...P.global.map(e => ({ kind: "g", idx: e.idx, label: featLabel("g", e.idx), e, color: "var(--s2)" })),
  ];
}
function metricValue(r, key, phase) {
  if (key.startsWith("gxi_") || key === "wscaled") {
    if (!r.kind || phase != null) return null;
    if (key === "wscaled") return r.kind === "s" ? DATA.importance.stem.spatial_scaled[r.idx] : DATA.importance.stem.global_scaled[r.idx];
    const g = DATA.importance.gradInput[key.slice(4)];
    return r.kind === "s" ? g.spatial[r.idx] : g.global[r.idx];
  }
  const src = phase == null ? r.e.all : r.e.phase[phase];
  return src ? src[key] : null;
}

function summaryCard() {
  const I = DATA.importance, m = DATA.meta;
  const rows = allFeatureRows();
  const best = kind => rows.filter(r => r.kind === kind).sort((a, b) => metricValue(b, "kl") - metricValue(a, "kl"))[0];
  const bestV = rows.slice().sort((a, b) => metricValue(b, "dwin") - metricValue(a, "dwin"))[0];
  const c = card("sec-summary", `Feature analysis · ${m.name}`, `${fmt.int(I.rows)} selfplay training rows from <code>${esc(m.tdataDir)}</code>, decoded exactly as the trainer does (distance channels S8–S11 from <code>spatialDistNCHW</code>). Generated ${esc(m.generated)}.`);
  const t = h("div", { class: "tiles" });
  t.append(tile("Base policy loss", fmt.fix(I.baseLoss.pol, 3), "vs 600-visit search targets"),
    tile("Base value loss", fmt.fix(I.baseLoss.val, 3), "vs game outcomes"),
    tile("Policy relies most on", best("s").label.replace(/^S\d+ · /, ""), `spatial · KL ${fmt.fix(metricValue(best("s"), "kl"), 2)}`),
    tile("Value relies most on", bestV.label.replace(/^[SG]\d+ · /, ""), `|ΔP(win)| ${fmt.pct(metricValue(bestV, "dwin"))}`));
  if (DATA.inputParity) { const ok = DATA.inputParity.maxAbsDiff < 1e-6; t.append(tile("Train = inference inputs", ok ? "✓" : "✗", `max |Δ| S8–S11 ${fmt.num(DATA.inputParity.maxAbsDiff)}`, ok ? null : "warn")); }
  if (DATA.mcts && DATA.mcts.summary) t.append(tile("Net top move = search", fmt.pct(DATA.mcts.summary.all.top1_nn, 0), `${DATA.mcts.summary.all.n} positions · ${m.visits} visits`));
  c.append(t);
  return c;
}

function parityCard() {
  const M = DATA.inputParity, F = DATA.features.spatial;
  const chs = M.channels.map(i => "S" + i).join(", ");
  const ok = M.maxAbsDiff < 1e-6;
  const c = card("sec-parity", "Input parity (S8–S11 distance planes)", `Channels ${chs} are continuous BFS distances /32. The training data stores them raw in <code>spatialDistNCHW</code> and the trainer decodes them to the same values <code>QuoridorNN::fillRow</code> feeds at inference (see <code>docs/DistPlanesUpgrade.md</code>). This check recomputes them from each row's pawn and blocked-edge planes with a Python replica of <code>fillRow</code> and compares.`, ok ? "" : "warn");
  const t = h("div", { class: "tiles" });
  t.append(tile("Max |trainer − replica|", fmt.num(M.maxAbsDiff), `S8–S11 over ${fmt.int(M.rows)} rows`, ok ? null : "warn"),
    tile("Rows identical", fmt.pct(M.rowsExact, 1), "S8–S11"),
    tile("Checkpoint migrated", M.migrated ? "yes" : "no", "zeroed stem weights for S8–S11 once"));
  if (DATA.mcts && DATA.mcts.summary) t.append(tile("Replica vs C++", fmt.num(DATA.mcts.summary.max_check_policy), "max |policy diff| on searched positions"));
  c.append(t);
  const tb = h("table", { class: "t", style: "max-width:760px" }, h("tr", {}, h("th", {}, "Plane"), h("th", {}, "mean value"), h("th", {}, "cells unreachable / ≥32"), h("th", {}, "conv_spatial |W|"), h("th", {}, "|W| / median other plane")));
  M.channels.forEach((ch, i) => tb.append(h("tr", {}, h("td", {}, `S${ch} ${F[ch].name}`), h("td", {}, fmt.fix(M.mean[i], 3)), h("td", {}, fmt.pct(M.unreachableFrac[i], 2)),
    h("td", {}, fmt.num(M.stemNorm[i])), h("td", {}, fmt.fix(M.stemNorm[i] / M.stemNormOtherMedian, 2)))));
  c.append(tb, h("div", { class: "note" }, "After the migration these first-layer weights restarted from zero; their norm relative to the other planes shows how far the net has re-learned to read the distances."));
  return c;
}

function importanceCard() {
  const c = card("sec-imp", "Feature importance", "Each feature (or group) is <b>permuted across positions</b> (constant planes are zeroed instead) and the net is re-run. Output-change metrics show how much the net <i>uses</i> the feature; loss metrics show whether that use is <i>correct</i> against search / outcome targets. Grouped features matter because many features are redundant (e.g. five encodings of the same fence count): permuting one alone underestimates the group.");
  const ctrl = h("div", { class: "ctrl" });
  const host = h("div", { class: "chart" }), desc = h("div", { class: "sub" });
  const draw = () => {
    host.innerHTML = "";
    const P = DATA.importance.permutation;
    const mdef = METRICS.find(m => m[0] === featState.metric) || [featState.metric, GXI.find(g => g[0] === featState.metric)[1], "Per-feature mean |∂output/∂x · (x − mean)| over positions (single features only).", x => fmt.num(x)];
    desc.textContent = mdef[2];
    let rows = featState.scope === "groups" ? P.groups.map(g => ({ label: g.name, e: g, color: "var(--s3)", group: g })) : allFeatureRows();
    rows = rows.map(r => ({ ...r, value: metricValue(r, featState.metric) })).filter(r => r.value != null).sort((a, b) => b.value - a.value);
    host.append(hbarChart(rows.map(r => ({ label: r.label + (r.e.mode === "zero" ? " (zeroed)" : ""), value: r.value, color: r.color,
      tip: () => impTip(r) })), { width: 760, labelWidth: 280, valFmt: mdef[3], tickFmt: mdef[3] }));
    host.append(featState.scope === "groups" ? legend([["Feature group", "var(--s3)"]]) : legend([["Spatial plane", "var(--s1)"], ["Global feature", "var(--s2)"]]));
  };
  ctrl.append("Metric", selectBox([...METRICS.map(m => [m[0], m[1]]), ...GXI], featState.metric, v => { featState.metric = v; if (v.startsWith("gxi") || v === "wscaled") { featState.scope = "single"; } draw(); }),
    segmented([["single", "Single features"], ["groups", "Groups"]], featState.scope, v => { featState.scope = v; draw(); }));
  c.append(ctrl, desc, host);
  draw();
  c.append(h("div", { class: "note" }, "Notes: S0 (on-board mask) is not ablated because the net uses it as its attention / pooling mask. Permuting pawns or walls alone creates impossible positions (planes that disagree with each other); large numbers there mean “the net reads this plane directly”."));
  return c;
}
function impTip(r) {
  const a = r.e.all;
  let s = `<b>${esc(r.label)}</b>${r.e.mode === "zero" ? " (zeroed)" : ""}<br>`;
  if (r.kind) s += esc(r.kind === "s" ? DATA.features.spatial[r.idx].desc : DATA.features.global[r.idx].desc) + "<br>";
  if (r.group) s += `planes ${r.group.spatial.map(i => "S" + i).join(" ") || "—"} · globals ${r.group.global.map(i => "G" + i).join(" ") || "—"}<br>`;
  s += METRICS.map(m => `${m[1]}: <b>${m[3](a[m[0]])}</b>`).join("<br>");
  return s;
}

function phaseCard() {
  const ph = DATA.features.phases;
  const c = card("sec-phase", "Importance by game phase", `Same permutation runs, split by walls already placed: ${ph.map(p => `${p.name} ${p.lo}–${p.hi}`).join(", ")}.`);
  const host = h("div", { class: "chart", style: "overflow-x:auto" });
  let metric = "kl", scope = "single";
  const draw = () => {
    host.innerHTML = "";
    const mdef = METRICS.find(m => m[0] === metric);
    let rows = scope === "groups" ? DATA.importance.permutation.groups.map(g => ({ label: g.name, e: g })) : allFeatureRows();
    rows = rows.map(r => ({ ...r, v: metricValue(r, metric) })).sort((a, b) => b.v - a.v);
    const m = rows.map(r => ph.map((_, i) => metricValue(r, metric, i)));
    host.append(heatTable(rows.map(r => r.label), ph.map((p, i) => `${p.name} (${fmt.si(DATA.importance.phaseCounts[i])})`), m,
      { cellW: 130, labelWidth: 280, valFmt: mdef[3], div: metric.startsWith("d") && metric !== "dwin" && metric !== "dmargin",
        tip: (i, j, v) => `<b>${esc(rows[i].label)}</b><br>${esc(ph[j].name)}: ${mdef[3](v)}` }));
  };
  c.append(h("div", { class: "ctrl" }, "Metric", selectBox(METRICS.map(m => [m[0], m[1]]), metric, v => { metric = v; draw(); }),
    segmented([["single", "Single features"], ["groups", "Groups"]], scope, v => { scope = v; draw(); })), host);
  draw();
  return c;
}

function stemCard() {
  const st = DATA.importance.stem;
  const c = card("sec-stem", "First-layer weights", "How strongly the first layer reads each input: the norm of <code>conv_spatial</code> weights per input plane and of <code>linear_global</code> columns per global feature, raw and multiplied by the input's RMS on real positions (≈ typical contribution to the embedding). Weight norms ignore everything downstream, so treat them as a sanity check next to the ablations.");
  const cols = h("div", { class: "cols2" });
  const mk = (kind, norms, scaled) => {
    const n = kind === "s" ? DATA.features.spatial.length : DATA.features.global.length;
    const rows = [];
    for (let i = 0; i < n; i++) rows.push({ label: featLabel(kind, i), value: scaled[i], raw: norms[i] });
    rows.sort((a, b) => b.value - a.value);
    return hbarChart(rows.map(r => ({ ...r, color: kind === "s" ? "var(--s1)" : "var(--s2)", tip: `<b>${esc(r.label)}</b><br>|W| ${fmt.num(r.raw)} · × RMS ${fmt.num(r.value)}` })), { width: 520, labelWidth: 230, rowH: 18 });
  };
  cols.append(h("div", { class: "chart" }, h("div", { class: "ttl" }, "Spatial planes: |W| × input RMS"), mk("s", st.spatial_norm, st.spatial_scaled)),
    h("div", { class: "chart" }, h("div", { class: "ttl" }, "Global features: |W| × input RMS"), mk("g", st.global_norm, st.global_scaled)));
  c.append(cols, h("div", { class: "note" }, "The 3×3 kernel footprint of every plane is in the Architecture tab (click conv_spatial)."));
  return c;
}

function calibrationCard() {
  const C = DATA.calibration;
  const c = card("sec-cal", "Calibration against game outcomes", `On the same training rows. Brier score ${fmt.fix(C.brier, 4)}; margin MAE ${fmt.fix(C.margin_mae, 2)} moves.`);
  const cols = h("div", { class: "cols2" });
  const vpts = C.value.filter(b => b.n > 0).map(b => [b.pred, b.actual, 3 + Math.min(6, Math.sqrt(b.n) / 6)]);
  cols.append(h("div", { class: "chart" }, h("div", { class: "ttl" }, "P(win): predicted vs actual"), h("div", { class: "sub" }, "Dots = 10% bins (size ∝ rows); line = perfect calibration."),
    lineChart([{ name: "actual win rate", color: "var(--s1)", pts: vpts, dots: true }], { xmin: 0, xmax: 1, ymin: 0, ymax: 1, diag: true, width: 420, height: 260, xFmt: x => fmt.pct(x, 0), yFmt: x => fmt.pct(x, 0), xName: "pred", xLabel: "predicted P(win)" })));
  const mp = C.margin.map(b => [b.pred, b.actual]);
  cols.append(h("div", { class: "chart" }, h("div", { class: "ttl" }, "Game margin: predicted vs actual"), h("div", { class: "sub" }, "Mean final margin per 2-move bin of the prediction; band = 10th–90th percentile."),
    lineChart([{ name: "actual margin", color: "var(--s3)", pts: mp, dots: true, band: C.margin.map(b => [b.pred, b.p10, b.p90]) }], { diag: true, width: 420, height: 260, xName: "pred", xLabel: "predicted margin (moves)", yFmt: x => fmt.fix(x, 1), xFmt: x => fmt.fix(x, 1),
      xmin: Math.min(...mp.map(p => p[0])), xmax: Math.max(...mp.map(p => p[0])), ymin: Math.min(...C.margin.map(b => b.p10)), ymax: Math.max(...C.margin.map(b => b.p90)) })));
  c.append(cols);
  return c;
}

function searchCard() {
  const S = DATA.mcts.summary, ph = DATA.features.phases;
  const c = card("sec-search", "Raw net vs C++ search", `${S.all.n} positions sampled from selfplay games, searched with <code>kata-search_analyze</code> at ${DATA.meta.visits} visits (GTP config). Python replica check: max |policy diff| vs C++ ${fmt.num(S.max_check_policy)}, max |P(win) diff| ${fmt.num(S.max_check_win)}.`);
  const t = h("div", { class: "tiles" });
  t.append(tile("Top move = search best", fmt.pct(S.all.top1_nn, 0)), tile("Prior of search's best", fmt.pct(S.all.prior_of_best, 0), "mean"),
    tile("KL(search ‖ net)", fmt.fix(S.all.kl_nn, 3)), tile("|P(win) − search|", fmt.pct(S.all.value_mae, 1), "mean"),
    tile("Net wall mass", fmt.pct(S.all.wall_mass_nn, 0), `search visits on walls ${fmt.pct(S.all.wall_share_mcts, 0)}`));
  c.append(t);
  const cols = h("div", { class: "cols2" });
  const cats = ph.map(p => p.name), sub = S.phase.map(p => p.n + " pos");
  cols.append(h("div", { class: "chart" }, h("div", { class: "ttl" }, "Agreement with search by phase"),
    groupedBars(cats, [{ name: "net top move = search best", color: "var(--s1)", vals: S.phase.map(p => p.top1_nn) }, { name: "mean prior of search best", color: "var(--s3)", vals: S.phase.map(p => p.prior_of_best) }], { yFmt: x => fmt.pct(x, 0), max: 1, sub }),
    legend([["net top move = search best", "var(--s1)"], ["mean prior of search best", "var(--s3)"]])));
  cols.append(h("div", { class: "chart" }, h("div", { class: "ttl" }, "Walls: net prior vs search"),
    groupedBars(cats, [{ name: "net policy mass on walls", color: "var(--s2)", vals: S.phase.map(p => p.wall_mass_nn) }, { name: "search visit share on walls", color: "var(--s7)", vals: S.phase.map(p => p.wall_share_mcts) }], { yFmt: x => fmt.pct(x, 0), sub }),
    legend([["net policy mass on walls", "var(--s2)"], ["search visit share on walls", "var(--s7)"]])));
  cols.append(h("div", { class: "chart" }, h("div", { class: "ttl" }, "Value error vs search by phase"),
    groupedBars(cats, [{ name: "|P(win) − search|", color: "var(--s4)", vals: S.phase.map(p => p.value_mae) }], { yFmt: x => fmt.pct(x, 0), sub })));
  c.append(cols);
  return c;
}

let browserSort = "kl", browserSel = 0, browserOverlay = "nn";
function browserCard() {
  const c = card("sec-browser", "Position browser", "Sampled positions with the raw net, the search, per-position attributions (grad × input) and the auxiliary heads. Hover moves for numbers.");
  const P = DATA.mcts.positions;
  const list = h("div", { class: "plist" }), boardBox = h("div", { class: "boardBox" }), info = h("div", {});
  const sorts = { kl: ["KL(search‖net)", p => -p.kl_mcts_nn, p => fmt.fix(p.kl_mcts_nn, 2)], prior: ["Low prior of best", p => p.prior_of_best, p => fmt.pct(p.prior_of_best, 0)],
    verr: ["Value error", p => -Math.abs(p.nn.win - p.mcts.win), p => fmt.pct(Math.abs(p.nn.win - p.mcts.win), 0)], ply: ["Game / ply", (p, i) => i, p => "ply " + p.ply] };
  const drawList = () => {
    list.innerHTML = "";
    const idx = P.map((p, i) => i).sort((a, b) => sorts[browserSort][1](P[a], a) - sorts[browserSort][1](P[b], b));
    for (const i of idx) {
      const p = P[i];
      const disagree = p.nn.top !== p.mcts.best;
      list.append(h("div", { class: "prow" + (i === browserSel ? " sel" : ""), onclick: () => { browserSel = i; drawList(); drawPos(); } },
        h("span", { class: "i" }, "#" + (i + 1)), h("span", { class: "m" }, `${p.state.toMove} · ${p.walls}w · ${disagree ? "≠ " : ""}${p.mcts.best}`), h("span", { class: "v" }, sorts[browserSort][2](p))));
    }
  };
  const overlays = [["nn", "Net policy"], ["mcts", "Search visits"], ["attrV", "Value attribution"], ["attrP", "Policy attribution"], ["traj", "Trajectory"], ["walls", "Final walls"]];
  const drawPos = () => {
    const p = P[browserSel]; if (!p) return;
    boardBox.innerHTML = "";
    const svg = svgRoot(10, 10), st = p.state, o = {};
    const pct = v => v >= 0.995 ? "99" : (v * 100).toFixed(v < 0.1 ? 1 : 0);
    if (["nn", "mcts"].includes(browserOverlay)) {
      const pol = browserOverlay === "nn" ? p.nn.policy : p.mcts.policy;
      const all = [...pol.pawn, ...pol.v, ...pol.h].filter(v => v != null);
      const max = Math.max(1e-9, ...all);
      o.cellHeat = { vals: pol.pawn, kind: "policy", max, labels: v => pct(v), labelMin: 0.02 };
      o.wallHeat = { v: pol.v, h: pol.h, max, color: "var(--s3)", };
      o.wallTip = (kind, cc, r, v) => `<b>${colName(cc)}${r + 1}${kind}</b> ${fmt.pct(v, 1)}`;
      o.wallLabels = [];
      for (const kind of ["v", "h"]) pol[kind].forEach((v, k) => { if (v != null && v >= 0.03) o.wallLabels.push({ kind, c: k % 8, r: Math.floor(k / 8), text: pct(v) + "%" }); });
    } else if (browserOverlay === "attrV" || browserOverlay === "attrP") {
      o.cellHeat = { vals: p.attr[browserOverlay === "attrV" ? "value" : "policy"].cells, kind: "div" };
    } else if (browserOverlay === "traj") {
      const me = p.traj.me, op = p.traj.opp;
      o.cellHeat = { vals: me.map((v, k) => v - op[k]), kind: "div", max: 1 };
    } else if (browserOverlay === "walls") {
      o.wallHeat = { v: p.finalWalls.v, h: p.finalWalls.h, max: 1, color: "var(--s5)" };
      o.wallTip = (kind, cc, r, v) => `<b>${colName(cc)}${r + 1}${kind}</b> final-wall prob ${fmt.pct(v, 0)}`;
    }
    o.marks = [];
    const mk = (mv, color, dash) => { if (!mv) return; const m = parseMove(mv); o.marks.push({ ...m, color, dash }); };
    mk(p.mcts.best, "#f5d58a"); if (p.nn.top !== p.mcts.best) mk(p.nn.top, "#ffffff", "4 3");
    drawQBoard(svg, st, o);
    boardBox.append(svg, h("div", { class: "caption" }, "Solid gold outline: search's best move. Dashed white: net's top move if different. ",
      browserOverlay === "traj" ? "Blue = my future path likelier, red = opponent's." : browserOverlay.startsWith("attr") ? "Blue raises, red lowers the objective (sum of grad × input over planes)." : ""));
    // info panel
    info.innerHTML = "";
    const kv = h("dl", { class: "kv" });
    const add = (k, v) => kv.append(h("dt", {}, k), h("dd", {}, v));
    add("Game", `${p.file} · ply ${p.ply}/${p.nMoves} · ${p.gtype} · result ${p.result}`);
    add("To move", `${p.state.toMove === "B" ? "Black" : "White"} · fences B ${p.state.fB} / W ${p.state.fW}`);
    add("P(win) mover", `net ${fmt.pct(p.nn.win)} · search ${fmt.pct(p.mcts.win)}`);
    add("Margin mover", `net ${fmt.fix(p.nn.margin, 2)} · search ${fmt.fix(p.mcts.margin, 2)}`);
    add("Top move", `net ${p.nn.top} · search ${p.mcts.best} · played ${p.played}`);
    add("KL(search‖net)", fmt.fix(p.kl_mcts_nn, 3));
    info.append(kv);
    const tb = h("table", { class: "t", style: "margin-top:10px" }, h("tr", {}, ...["Move", "visits", "prior", "win", "margin", "PV"].map(x => h("th", {}, x))));
    for (const m of p.mcts.moves) tb.append(h("tr", { class: m.move === p.mcts.best ? "hl" : null }, h("td", { class: "mv" }, m.move), h("td", {}, m.visits), h("td", {}, fmt.pct(m.prior, 1)), h("td", {}, fmt.pct(m.win, 1)), h("td", {}, fmt.fix(m.margin, 1)), h("td", { class: "mv", style: "text-align:left" }, (m.pv || []).join(" "))));
    info.append(tb);
    const key = browserOverlay === "attrP" ? "policy" : "value";
    const at = p.attr[key];
    const items = [...at.spatial.map((v, i) => ({ label: featLabel("s", i), value: v, color: "var(--s1)" })), ...at.global.map((v, i) => ({ label: featLabel("g", i), value: v, color: "var(--s2)" }))]
      .filter(x => Math.abs(x.value) > 1e-4).sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 14);
    info.append(h("h3", {}, key === "value" ? "What drives this value (grad × input)" : "What drives the top move (grad × input)"),
      hbarChart(items, { width: 520, labelWidth: 230, rowH: 18, valFmt: x => fmt.sgn(x, 3) }));
  };
  c.append(h("div", { class: "ctrl" }, "Sort", selectBox(Object.entries(sorts).map(([k, v]) => [k, v[0]]), browserSort, v => { browserSort = v; drawList(); }),
    "Overlay", segmented(overlays, browserOverlay, v => { browserOverlay = v; drawPos(); })));
  c.append(h("div", { class: "browser" }, list, boardBox, info));
  drawList(); drawPos();
  return c;
}
