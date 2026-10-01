"use strict";
// ---- Playback --------------------------------------------------------------
function stopPlay() { if (playTimer) { clearInterval(playTimer); playTimer = null; $("bPlay").textContent = "Play"; } }
function togglePlay() {
  if (!cur) return;
  if (playTimer) return stopPlay();
  if (ply >= cur.moves.length) ply = cur.startTurnIdx;
  $("bPlay").textContent = "Pause";
  playTimer = setInterval(() => { if (ply >= cur.moves.length) return stopPlay(); ply++; update(); }, 450);
}
const go = d => { if (cur) { stopPlay(); ply += d; update(); } };

// ---- Events ----------------------------------------------------------------
function setTab(stats) {
  document.body.classList.toggle("stats", stats);
  $("tabStats").classList.toggle("on", stats);
  $("tabView").classList.toggle("on", !stats);
  $("flipBtn").hidden = stats;
  hideTip();
  if (stats) { stopPlay(); renderStats(); } else if (cur) renderChart();
}
$("tabView").onclick = () => setTab(false);
$("tabStats").onclick = () => setTab(true);
$("sType").addEventListener("input", renderStats);
$("fileInput").addEventListener("change", e => { if (e.target.files[0]) loadFile(e.target.files[0]); });
$("flipBtn").onclick = () => { flipped = !flipped; update(false); };
$("pathBtn").onclick = () => { showPaths = !showPaths; $("pathBtn").classList.toggle("on", showPaths); update(false); };
initSortSelect();
$("sortKey").addEventListener("input", e => setSort(e.target.value));
$("sortDir").onclick = () => { sortDesc = !sortDesc; applyFilters(); };
$("randBtn").onclick = randomGame;
$("bFirst").onclick = () => { if (cur) { stopPlay(); ply = 0; update(); } };
$("bLast").onclick = () => { if (cur) { stopPlay(); ply = cur.moves.length; update(); } };
$("bPrev").onclick = () => go(-1);
$("bNext").onclick = () => go(1);
$("bPrev10").onclick = () => go(-10);
$("bNext10").onclick = () => go(10);
$("bPlay").onclick = togglePlay;
$("moveSlider").oninput = e => { stopPlay(); ply = +e.target.value; update(); };
for (const id of ["fRes", "fType", "fMin"]) $(id).addEventListener("input", applyFilters);
$("gameList").addEventListener("click", e => { const r = e.target.closest(".g"); if (r) selectGame(+r.dataset.g); });
$("moveList").addEventListener("click", e => { const r = e.target.closest(".m"); if (r) { stopPlay(); ply = +r.dataset.i; update(false); } });

// wheel on the board: down = next move, up = previous. Accumulate so trackpads don't race.
let wheelAcc = 0;
$("boardWrap").addEventListener("wheel", e => {
  if (!cur) return;
  e.preventDefault();
  const unit = e.deltaMode === 1 ? 1 : e.deltaMode === 2 ? 3 : 1 / 60;   // lines / pages / pixels
  wheelAcc += e.deltaY * unit;
  const steps = Math.trunc(wheelAcc);
  if (steps) { wheelAcc -= steps; go(steps > 0 ? 1 : -1); }
}, { passive: false });

// chart: click to jump, hover crosshair + tooltip
$("chart").addEventListener("click", e => { if (cur) { stopPlay(); ply = chartPlyAt(e); update(); } });
$("chart").addEventListener("mousemove", e => {
  if (!cur) return;
  const i = chartPlyAt(e), { X } = chartGeom(), hv = $("chart")._hover;
  if (hv) { hv.setAttribute("x1", X(i)); hv.setAttribute("x2", X(i)); hv.setAttribute("visibility", "visible"); }
  const mv = cur.moves[i];
  let html = `<b>Position ${i}</b>` + (i < cur.moves.length ? ` · next ${mv.pla} ${notation(mv)}` : " · end");
  if (hasEv(mv)) html += `<br>${sw("var(--w-acc)")}White win ${pct(mv.ev.wWin)}<br>${sw("var(--margin)")}${mv.ev.lead !== undefined ? `Lead ${fmtScore(mv.ev.lead)} · utility ${fmtScore(mv.ev.wScore)}` : `Margin ${fmtScore(mv.ev.wScore)}`}<br>visits ${mv.ev.v ?? "?"}`;
  else html += `<br>${i < cur.startTurnIdx ? "random init" : i >= cur.moves.length ? games[curGameIdx].result : "no search"}`;
  showTip(e, html);
});
$("chart").addEventListener("mouseleave", () => { hideTip(); const hv = $("chart")._hover; if (hv) hv.setAttribute("visibility", "hidden"); });

window.addEventListener("resize", () => { if (cur && !document.body.classList.contains("stats")) renderChart(); });
document.addEventListener("keydown", e => {
  if (document.body.classList.contains("stats")) return;
  if (e.target.tagName === "INPUT" && e.target.type !== "range" || e.target.tagName === "SELECT") return;
  const k = e.key;
  if (k === "ArrowRight") go(1);
  else if (k === "ArrowLeft") go(-1);
  else if (k === "PageDown") go(10);
  else if (k === "PageUp") go(-10);
  else if (k === "Home") $("bFirst").click();
  else if (k === "End") $("bLast").click();
  else if (k === "ArrowDown") stepGame(1);
  else if (k === "ArrowUp") stepGame(-1);
  else if (k === " ") togglePlay();
  else if (k === "f" || k === "F") $("flipBtn").click();
  else if (k === "p" || k === "P") $("pathBtn").click();
  else if (k === "r" || k === "R") randomGame();
  else return;
  e.preventDefault();
});
["dragenter", "dragover"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); document.body.classList.add("dragover"); }));
["dragleave", "drop"].forEach(t => document.addEventListener(t, e => { e.preventDefault(); document.body.classList.remove("dragover"); }));
document.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });

// Served by serve.py: ?src=<url> loads that file directly.
const params = new URLSearchParams(location.search);
const src = params.get("src");
if (src) {
  $("fileName").textContent = "Loading…";
  fetch(src).then(r => { if (!r.ok) throw new Error(r.status); return r.text(); })
    .then(t => loadText(t, params.get("name") || src))
    .catch(err => { $("fileName").textContent = `Failed to load ${src}: ${err}`; });
}
