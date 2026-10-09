"use strict";
const go = d => { if (cur) { ply += d; update(); } };
function setTab(stats) {
  document.body.classList.toggle("stats", stats);
  $("tabStats").classList.toggle("on", stats);
  $("tabView").classList.toggle("on", !stats);
  $("flipBtn").hidden = stats; $("pathBtn").hidden = stats;
  hideTip();
  if (stats) renderStats(); else if (cur) update(false);
}
$("tabView").onclick = () => setTab(false);
$("tabStats").onclick = () => setTab(true);
$("fileInput").addEventListener("change", e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
$("flipBtn").onclick = () => { rot = (rot + 1) % 4; if (cur) update(false); };
$("pathBtn").onclick = () => { showPaths = !showPaths; $("pathBtn").classList.toggle("on", showPaths); if (cur) update(false); };
initSortSelect();
$("sortKey").addEventListener("input", e => setSort(e.target.value));
$("sortDir").onclick = () => { sortDesc = !sortDesc; applyFilters(true); };
$("randBtn").onclick = randomGame;
$("bFirst").onclick = () => { if (cur) { ply = 0; update(); } };
$("bLast").onclick = () => { if (cur) { ply = cur.events.length; update(); } };
$("bPrev").onclick = () => go(-1);
$("bNext").onclick = () => go(1);
$("bPrev10").onclick = () => go(-10);
$("bNext10").onclick = () => go(10);
$("moveSlider").oninput = e => { ply = +e.target.value; update(); };
for (const id of FILTER_IDS) $(id).addEventListener("input", () => applyFilters());
$("fChips").addEventListener("click", e => { if (e.target.closest("button")) clearFilters(); });

{
  const LIST_W_KEY = "q4SgfsViewer.listWidth", main = document.querySelector("main"), handle = $("listResize");
  const setW = w => { w = Math.round(Math.max(220, Math.min(w, innerWidth - 640))); main.style.setProperty("--listW", w + "px"); return w; };
  try { const w = +localStorage.getItem(LIST_W_KEY); if (w) setW(w); } catch (e) { /* storage unavailable */ }
  handle.addEventListener("pointerdown", e => {
    e.preventDefault();
    handle.setPointerCapture(e.pointerId);
    const x0 = e.clientX, w0 = $("gamesPanel").getBoundingClientRect().width;
    document.body.classList.add("resizing");
    const move = ev => setW(w0 + ev.clientX - x0);
    const up = ev => {
      handle.removeEventListener("pointermove", move); handle.removeEventListener("pointerup", up); handle.removeEventListener("pointercancel", up);
      document.body.classList.remove("resizing");
      try { localStorage.setItem(LIST_W_KEY, setW(w0 + ev.clientX - x0)); } catch (err) { /* storage unavailable */ }
      if (cur) renderChart();
    };
    handle.addEventListener("pointermove", move); handle.addEventListener("pointerup", up); handle.addEventListener("pointercancel", up);
  });
  handle.addEventListener("dblclick", () => { main.style.removeProperty("--listW"); try { localStorage.removeItem(LIST_W_KEY); } catch (e) { /* storage unavailable */ } if (cur) renderChart(); });
}
$("gameList").addEventListener("click", e => { const r = e.target.closest(".g"); if (r) selectGame(+r.dataset.g); });
$("moveList").addEventListener("click", e => { const r = e.target.closest(".m"); if (r) { ply = +r.dataset.i; update(false); } });

let wheelAcc = 0;
$("boardWrap").addEventListener("wheel", e => {
  if (!cur) return;
  e.preventDefault();
  const unit = e.deltaMode === 1 ? 1 : e.deltaMode === 2 ? 3 : 1 / 60;
  wheelAcc += e.deltaY * unit;
  const steps = Math.trunc(wheelAcc);
  if (steps) { wheelAcc -= steps; go(steps > 0 ? 1 : -1); }
}, { passive: false });

const narrow = window.matchMedia("(max-width: 1100px)");
$("gamesHead").addEventListener("click", () => { if (narrow.matches) $("gamesPanel").classList.toggle("open"); });
$("gameList").addEventListener("click", () => { if (narrow.matches) $("gamesPanel").classList.remove("open"); });
{
  let x0 = null, y0 = 0;
  const wrap = $("boardWrap");
  wrap.addEventListener("pointerdown", e => { if (e.pointerType === "touch") { x0 = e.clientX; y0 = e.clientY; } });
  wrap.addEventListener("pointerup", e => {
    if (x0 === null || !cur) return;
    const dx = e.clientX - x0, dy = e.clientY - y0;
    x0 = null;
    if (Math.abs(dx) > 40 && Math.abs(dx) > 1.5 * Math.abs(dy)) go(dx < 0 ? 1 : -1);
  });
  wrap.addEventListener("pointercancel", () => { x0 = null; });
}

$("chart").addEventListener("click", e => { if (cur) { ply = chartPlyAt(e); update(); } });
$("chart").addEventListener("mousemove", e => {
  if (!cur) return;
  const i = chartPlyAt(e), g = chartGeom(), hv = $("chart")._hover;
  if (hv) { hv.setAttribute("x1", g.X(i)); hv.setAttribute("x2", g.X(i)); hv.setAttribute("visibility", "visible"); }
  const ev = cur.events[i];
  let html = `<b>Position ${i}</b>` + (ev ? ` · next ${SEATS[ev.seat]} ${ev.elim ? "eliminated" : ev.text}` : " · end");
  if (ev && ev.comment) {
    const names = ["S", "W", "N", "E", "draw"], cols = SEAT_COL.concat([DRAW_COL]);
    html += ev.comment.p.map((p, t) => `<br>${sw(cols[t])}${names[t]} ${pct(p, 1)}`).join("") + `<br>visits ${ev.comment.visits}`;
  } else html += `<br>${ev && i < cur.h.startTurnIdx ? "opening ply" : ev ? "no search" : resultFull(cur.h)}`;
  showTip(e, html);
});
$("chart").addEventListener("mouseleave", () => { hideTip(); const hv = $("chart")._hover; if (hv) hv.setAttribute("visibility", "hidden"); });

window.addEventListener("resize", () => { if (cur && !document.body.classList.contains("stats")) renderChart(); });
document.addEventListener("keydown", e => {
  if (document.body.classList.contains("stats")) return;
  if (e.target.tagName === "INPUT" && e.target.type !== "range" || e.target.tagName === "SELECT") return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key;
  if (k === "ArrowRight") go(1);
  else if (k === "ArrowLeft") go(-1);
  else if (k === "PageDown") go(10);
  else if (k === "PageUp") go(-10);
  else if (k === "Home") $("bFirst").click();
  else if (k === "End") $("bLast").click();
  else if (k === "ArrowDown") stepGame(1);
  else if (k === "ArrowUp") stepGame(-1);
  else if (k === "r" || k === "R") $("flipBtn").click();
  else if (k === "p" || k === "P") $("pathBtn").click();
  else return;
  e.preventDefault();
});
["dragenter", "dragover"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); document.body.classList.add("dragover"); }));
["dragleave", "drop"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); document.body.classList.remove("dragover"); }));
document.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

const params = new URLSearchParams(location.search);
const src = params.get("src");
if (src) {
  $("fileName").textContent = "Loading…";
  fetch(src).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(t => loadText(t, params.get("name") || src))
    .catch(err => { $("fileName").textContent = `Failed to load ${src}: ${err}`; });
}
