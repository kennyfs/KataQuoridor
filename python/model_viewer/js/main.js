"use strict";
// ---- loading & tabs ------------------------------------------------------------------------------------------
let DATA = null;
let tab = "arch";

function setTab(t) {
  tab = t;
  $("tabArch").classList.toggle("on", t === "arch");
  $("tabFeat").classList.toggle("on", t === "feat");
  if (!DATA) return;
  $("archView").hidden = t !== "arch";
  $("featView").hidden = t !== "feat";
}

function loadData(d, label) {
  DATA = d;
  _paramCache.clear(); _attnCache.clear();
  $("empty").hidden = true;
  const m = d.meta;
  $("metaLine").textContent = `${label || m.name} · ${fmt.si(m.numParams)} params · ${fmt.si(m.trainState.global_step_samples)} samples · generated ${m.generated}`;
  renderArch();
  renderAnalysis();
  setTab(tab);
  $("detail").innerHTML = '<div class="hintBox">Click any layer to see its weights, statistics and (for attention layers) attention maps.</div>';
}

async function fetchJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + " " + r.status); return r.json(); }

async function init() {
  $("tabArch").onclick = () => setTab("arch");
  $("tabFeat").onclick = () => setTab("feat");
  addEventListener("keydown", ev => { if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return; if (ev.key === "1") setTab("arch"); if (ev.key === "2") setTab("feat"); });
  $("fileInput").onchange = ev => { const f = ev.target.files[0]; if (f) readFile(f); };
  addEventListener("dragover", ev => { ev.preventDefault(); document.body.classList.add("dragover"); });
  addEventListener("dragleave", () => document.body.classList.remove("dragover"));
  addEventListener("drop", ev => { ev.preventDefault(); document.body.classList.remove("dragover"); const f = ev.dataTransfer.files[0]; if (f) readFile(f); });
  const sel = $("modelSel");
  sel.hidden = true;
  try {
    const idx = await fetchJSON("data/index.json");
    if (idx.length) {
      sel.hidden = false;
      for (const e of idx) sel.append(h("option", { value: e.file }, e.name));
      const want = new URLSearchParams(location.search).get("model");
      sel.value = idx.find(e => e.name === want || e.file === want) ? (idx.find(e => e.name === want || e.file === want).file) : idx[idx.length - 1].file;
      sel.onchange = () => loadFile(sel.value);
      await loadFile(sel.value);
    }
  } catch (e) { console.warn("no data/index.json", e); }
}

async function loadFile(file) {
  $("metaLine").textContent = "Loading " + file + " …";
  loadData(await fetchJSON("data/" + file));
}

function readFile(f) {
  const rd = new FileReader();
  rd.onload = () => { try { loadData(JSON.parse(rd.result), f.name); } catch (e) { alert("Not an analysis file: " + e.message); } };
  rd.readAsText(f);
}

init();
