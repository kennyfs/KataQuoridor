"use strict";
// ---- DOM helpers -------------------------------------------------------------
const $ = id => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";

// h("div", {class: "x", onclick: fn, style: "..."}, child, "text", ...)
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c instanceof Node ? c : String(c));
  return e;
}
function sv(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v !== null && v !== undefined) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function svText(parent, x, y, s, attrs) { const t = sv("text", { x, y, ...attrs }, parent); t.textContent = s; return t; }
function svgRoot(w, hgt, cls) { return sv("svg", { viewBox: `0 0 ${w} ${hgt}`, class: cls || null, xmlns: SVGNS }); }

// ---- formatting ----------------------------------------------------------------
const fmt = {
  int: x => x == null ? "—" : Math.round(x).toLocaleString("en-US"),
  si: x => x == null ? "—" : Math.abs(x) >= 1e6 ? (x / 1e6).toFixed(2) + "M" : Math.abs(x) >= 1e3 ? (x / 1e3).toFixed(1) + "k" : String(Math.round(x)),
  pct: (x, d = 1) => x == null ? "—" : (x * 100).toFixed(d) + "%",
  num: (x, d = 3) => x == null ? "—" : Math.abs(x) >= 1000 ? x.toFixed(0) : Math.abs(x) >= 100 ? x.toFixed(1) : Number(x).toPrecision(d).replace(/\.?0+e/, "e"),
  fix: (x, d = 3) => x == null ? "—" : Number(x).toFixed(d),
  sgn: (x, d = 3) => x == null ? "—" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d),
  shape: s => s ? s.join("×") : "",
};
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---- base64 tensors ------------------------------------------------------------------
function b64Bytes(str) {
  const bin = atob(str), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
const _paramCache = new Map();
function paramValues(name) {   // dequantized Float32Array of a parameter
  if (_paramCache.has(name)) return _paramCache.get(name);
  const p = DATA.params[name];
  if (!p || !p.q) return null;
  const b = new Int8Array(b64Bytes(p.q).buffer), out = new Float32Array(b.length);
  for (let i = 0; i < b.length; i++) out[i] = b[i] * p.qscale;
  _paramCache.set(name, out);
  return out;
}

// ---- colors -----------------------------------------------------------------------
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function hex2rgb(h) { h = h.replace("#", ""); return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)]; }
function lerpRamp(ramp, t) {
  t = Math.max(0, Math.min(1, t));
  const x = t * (ramp.length - 1), i = Math.min(ramp.length - 2, Math.floor(x)), f = x - i;
  const a = ramp[i], b = ramp[i + 1];
  return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * f));
}
const rgbStr = c => `rgb(${c[0]},${c[1]},${c[2]})`;
// Sequential (blue, light -> dark) and diverging (red <- neutral -> blue), per references/palette.md
const SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"].map(hex2rgb);
const DIV_BLUE = ["#2a78d6", "#1c5cab", "#0d366b"].map(hex2rgb);
const DIV_RED = ["#e34948", "#b8302f", "#7d1d1c"].map(hex2rgb);
function isDark() { return document.documentElement.getAttribute("data-theme") === "dark" || (document.documentElement.getAttribute("data-theme") !== "light" && matchMedia("(prefers-color-scheme: dark)").matches); }
function neutralRGB() { return hex2rgb(isDark() ? "#383835" : "#f0efec"); }
function seqColor(t) {   // t in [0,1]; on dark the ramp starts from the surface-ish end
  const ramp = isDark() ? [hex2rgb("#1f2a38"), ...SEQ_BLUE.slice(5).map(x => x)] : [hex2rgb("#f4f8fd"), ...SEQ_BLUE];
  return lerpRamp(ramp, t);
}
function divColor(t) {   // t in [-1,1]: negative red, positive blue, 0 neutral gray
  const n = neutralRGB();
  if (t >= 0) return lerpRamp([n, [158, 197, 244], ...DIV_BLUE], Math.pow(t, 0.8));
  return lerpRamp([n, [245, 170, 168], ...DIV_RED], Math.pow(-t, 0.8));
}

// ---- tooltip --------------------------------------------------------------------------
function showTip(ev, html) {
  const t = $("tip");
  t.innerHTML = html;
  t.style.display = "block";
  const pad = 14, w = t.offsetWidth, hh = t.offsetHeight;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  if (x + w > innerWidth - 8) x = ev.clientX - w - pad;
  if (y + hh > innerHeight - 8) y = ev.clientY - hh - pad;
  t.style.left = x + "px"; t.style.top = y + "px";
}
function hideTip() { $("tip").style.display = "none"; }
function tipOn(elem, htmlFn) {
  elem.addEventListener("mousemove", ev => showTip(ev, typeof htmlFn === "function" ? htmlFn(ev) : htmlFn));
  elem.addEventListener("mouseleave", hideTip);
}

// segmented control: opts = [[value, label], ...]
function segmented(opts, cur, onChange) {
  const box = h("div", { class: "seg" });
  for (const [v, lab, title] of opts) {
    const b = h("button", { class: v === cur ? "on" : null, title: title || null, onclick: () => {
      for (const x of box.children) x.classList.remove("on");
      b.classList.add("on"); onChange(v);
    } }, lab);
    box.append(b);
  }
  return box;
}
function selectBox(opts, cur, onChange) {
  const s = h("select", { onchange: () => onChange(s.value) });
  for (const [v, lab] of opts) s.append(h("option", { value: v, selected: v === cur ? true : null }, lab));
  return s;
}
function tile(k, v, s, cls) { return h("div", { class: "tile " + (cls || "") }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), s ? h("div", { class: "s" }, s) : null); }

// Feature naming helpers (filled once DATA is loaded)
function featName(kind, idx) { return kind === "s" ? DATA.features.spatial[idx].name : DATA.features.global[idx].name; }
function featLabel(kind, idx) { return (kind === "s" ? "S" : "G") + idx + " · " + featName(kind, idx); }
