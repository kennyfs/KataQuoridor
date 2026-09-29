"use strict";
// ---- SVG helpers -----------------------------------------------------------
function el(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, attrs = {}) {
  const e = el("text", { x, y, "font-size": 10.5, fill: "#8a96a3", ...attrs }, parent);
  e.textContent = s;
  return e;
}
// bar with rounded top (4px) anchored flat on the baseline
function barPath(x, y, w, h, r = 4) {
  if (h <= 0) return "";
  r = Math.min(r, w / 2, h);
  return `M${x} ${y + h}V${y + r}Q${x} ${y} ${x + r} ${y}H${x + w - r}Q${x + w} ${y} ${x + w} ${y + r}V${y + h}Z`;
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function detectMatch() {
  const names = new Set();
  for (const g of games) if (g.pb !== g.pw) { names.add(g.pb); names.add(g.pw); }
  if (names.size !== 2) return null;
  const samples = n => { const m = /-s(\d+)/.exec(n); return m ? +m[1] : null; };
  let models = [...names].sort();
  const known = models.every(n => samples(n) !== null);
  if (known) models.sort((a, b) => samples(b) - samples(a));
  return { models, labels: known ? ["New", "Old"] : ["Model 1", "Model 2"], colors: ["var(--m-new)", "var(--m-old)"] };
}
const mIdx = name => match ? match.models.indexOf(name) : -1;
const mLabel = name => match && mIdx(name) >= 0 ? match.labels[mIdx(name)] : "";
const mDot = name => match && mIdx(name) >= 0 ? `<span class="mdot" style="background:${match.colors[mIdx(name)]}"></span>` : "";
const shortName = n => n.replace(/^run\d+-/, "");
const fmtScore = s => (s > 0 ? "+" : s < 0 ? "−" : "±") + Math.abs(s).toFixed(1);
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";

// tooltip
const tip = $("tip");
function showTip(e, html) {
  tip.innerHTML = html; tip.style.display = "block";
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 6) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 6) y = e.clientY - r.height - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.style.display = "none"; }
const sw = c => `<span class="sw" style="background:${c}"></span>`;

