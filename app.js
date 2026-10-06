// LMBuild project page. Reads data/*.json; every section degrades to empty if a file is missing.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const getJSON = u => fetch(u).then(r => r.ok ? r.json() : null).catch(() => null);

// ---- model identity -------------------------------------------------------
const NAME = {"gpt-6-astra": "GPT-6 Astra", "gpt-5.6-sol": "GPT-5.6 Sol", "claude-opus-5": "Claude Opus 5", "claude-fable-5.1": "Claude Fable 5.1",
  "claude-sonnet-5": "Claude Sonnet 5", "claude-haiku-4.5": "Claude Haiku 4.5", "qwen3.5-27b": "Qwen3.5-27B", "qwen3.5-35b-a3b": "Qwen3.5-35B-A3B",
  "gpt-oss-120b": "GPT-OSS-120B", "gemma-4-31b-it": "Gemma-4-31B-IT", "ministral-3-8b-instruct-2512": "Ministral-3-8B", "internvl3.5-38b": "InternVL3.5-38B",
  "qwen3-vl-30b-a3b-instruct": "Qwen3-VL-30B-A3B", "minicpm-v-4.5": "MiniCPM-V-4.5", "qwen3-4b-instruct-2507": "Qwen3-4B-Instruct",
  "qwen3-vl-32b-instruct": "Qwen3-VL-32B", "ernie-4.5-vl-28b-a3b": "ERNIE-4.5-VL-28B-A3B", "internvl3.5-8b": "InternVL3.5-8B",
  "qwen3-vl-8b-instruct": "Qwen3-VL-8B", "brickgpt": "BrickGPT", "legoace": "LegoACE", "partcrafter_np15": "PartCrafter-15",
  "partcrafter_np8": "PartCrafter-8", "partpacker": "PartPacker", "cubepart_core": "Cube3D + CubePart",
  "particulate-partcrafter": "PartCrafter-15† (P)", "particulate-partcrafter-np8": "PartCrafter-8† (P)",
  "particulate-partpacker": "PartPacker† (P)", "particulate-cube3d": "Cube3D† (P)", "physx-anything": "PhysX-Anything"};
const ORDER = {
  closed: ["gpt-6-astra", "gpt-5.6-sol", "claude-opus-5", "claude-fable-5.1", "claude-sonnet-5", "claude-haiku-4.5"],
  open: ["qwen3.5-27b", "qwen3.5-35b-a3b", "gpt-oss-120b", "gemma-4-31b-it", "ministral-3-8b-instruct-2512", "internvl3.5-38b",
    "qwen3-vl-30b-a3b-instruct", "minicpm-v-4.5", "qwen3-4b-instruct-2507", "qwen3-vl-32b-instruct", "ernie-4.5-vl-28b-a3b",
    "internvl3.5-8b", "qwen3-vl-8b-instruct"],
  ext: ["brickgpt", "legoace", "partcrafter_np15", "partcrafter_np8", "partpacker", "cubepart_core", "particulate-partcrafter",
    "particulate-partcrafter-np8", "particulate-partpacker", "particulate-cube3d", "physx-anything"]};
const GROUP = Object.fromEntries(Object.entries(ORDER).flatMap(([g, ids]) => ids.map(id => [id, g])));
const BY_NAME = Object.fromEntries(Object.entries(NAME).map(([k, v]) => [v, k]));
const vendor = id => GROUP[id] === "ext" ? "ext" : GROUP[id] === "open" ? "open" : id.startsWith("claude") ? "anthropic" : id.startsWith("gpt-") && GROUP[id] === "closed" ? "openai" : "open";
const LOGO = [[/^gpt-/, "openai.svg"], [/^claude/, "claude-color.svg"], [/^qwen/, "qwen-color.svg"], [/^gemma/, "gemma-color.svg"],
  [/^ministral/, "mistral-color.svg"], [/^internvl/, "internlm-color.svg"], [/^minicpm/, "openbmb.png"], [/^ernie/, "wenxin-color.svg"],
  [/partpacker/, "nvidia-color.svg"], [/cube/, "roblox.svg"]];
const CUBE = `<span class="glyph" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/></svg></span>`;
const logo = id => { const m = LOGO.find(([re]) => re.test(id)); return m ? `<img class="mlogo${m[1].endsWith(".png") ? " sq" : ""}" src="assets/logos/${m[1]}" alt="" loading="lazy">` : CUBE; };
const who = id => `${logo(id)}${esc(NAME[id] || id)}`;
const whoByName = n => BY_NAME[n] ? who(BY_NAME[n]) : esc(n);

const PLAB = {"baseline": "retrieve + create", "retrieval-only": "retrieve only", "creation-only": "create only"};
const GORD = {product: 0, sota: 1, open: 2};
const byGroup = xs => [...xs].sort((a, b) => GORD[a.group || "sota"] - GORD[b.group || "sota"]);
const tags = m => (m.protocol ? `<span class="tag ${m.protocol}">${PLAB[m.protocol] || m.protocol}</span>` : "") +
  (m.group === "product" ? ` <span class="tag product">real product</span>` : "");
const share = m => m.created_share == null ? "" : m.created_share <= 0.05 ? "all parts retrieved from the catalog"
  : m.created_share >= 0.95 ? "all parts created by the agent" : `${Math.round(100 * m.created_share)}% of parts created`;
const cap = s => s ? s[0].toUpperCase() + s.slice(1) : s;

// ---- lightbox ---------------------------------------------------------------
const lb = $("#lightbox");
function zoom(src, text) { lb.querySelector("img").src = src; lb.querySelector("figcaption").textContent = text || ""; lb.hidden = false; }
lb.onclick = () => { lb.hidden = true; lb.querySelector("img").src = ""; };
document.addEventListener("keydown", e => { if (e.key === "Escape" && !lb.hidden) lb.onclick(); });
document.addEventListener("click", e => {
  const z = e.target.closest("[data-zoom]");
  if (z) { e.preventDefault(); zoom(z.dataset.zoom, z.dataset.zcap); }
});

// ---- input -> output card --------------------------------------------------
let INPUTS = {};
function inputStrip(task) {
  const i = INPUTS[task];
  if (!i) return "";
  return `<div class="io-in"><img src="${i.img}" alt="Input image: ${esc(i.name)}" loading="lazy" data-zoom="${i.full || i.img}" data-zcap="Input image (${esc(i.image_id)})">
    <p class="q"><span class="lab">Input</span>“${esc(i.prompt)}”</p></div>`;
}
function ioCard(m, {imgClass = "", extra = ""} = {}) {
  const sub = m.joint ? ` · ${esc(m.joint_label || m.joint.replace(/_/g, " "))}` : "";
  return `<article class="card io">${inputStrip(m.task)}
    <div class="io-arrow"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6" stroke-linecap="round" stroke-linejoin="round"/></svg>Output</div>
    <div class="io-out"><img class="${imgClass}" src="${m.file || m.path}" alt="${esc(m.object)} built by ${esc(m.system)}" loading="lazy"></div>
    <div class="io-cap"><span class="who">${whoByName(m.system)}</span>${esc(cap(m.object))}${sub} ${tags(m)}${extra}</div></article>`;
}

// ---- section nav: highlight the section in view ---------------------------
(() => {
  const links = $$("#pnav a"), ids = links.map(a => a.dataset.sec);
  const on = id => links.forEach(a => a.classList.toggle("on", a.dataset.sec === id));
  const io = new IntersectionObserver(es => {
    const vis = es.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (vis.length) on(vis[0].target.id);
  }, {rootMargin: "-35% 0px -55% 0px"});
  ids.forEach(id => { const el = document.getElementById(id); if (el) io.observe(el); });
  on("overview");
})();

// ---- citation copy ------------------------------------------------------------
$("#copy-bib").onclick = async () => {
  const t = $("#copy-bib span");
  try { await navigator.clipboard.writeText($("#bib").textContent); t.textContent = "Copied"; }
  catch { t.textContent = "Select and copy"; }
  setTimeout(() => t.textContent = "Copy", 1600);
};

// ---- benchmark switch, linkable by hash -----------------------------------
const TABS = ["compare", "steps", "function", "data"];
function showTab(tab, scroll) {
  $$("#switch .sw").forEach(b => b.setAttribute("aria-selected", b.dataset.tab === tab));
  TABS.forEach(t => $(`#pane-${t}`).hidden = t !== tab);
  if (scroll) $("#switch").scrollIntoView({behavior: "smooth", block: "start"});
}
$$("#switch .sw").forEach(b => b.onclick = () => { showTab(b.dataset.tab); history.replaceState(null, "", "#" + b.dataset.tab); });
$$("[data-goto]").forEach(a => a.onclick = e => { e.preventDefault(); showTab(a.dataset.goto, true); });
if (TABS.includes(location.hash.slice(1))) showTab(location.hash.slice(1), true);

function segment(el, cb) {
  $$("button", el).forEach(b => b.onclick = () => { $$("button", el).forEach(x => x.setAttribute("aria-pressed", x === b)); cb(b.dataset.v || b.dataset.g); });
}

// ---- leaderboard -------------------------------------------------------------
const LEVELS = [
  {key: "Soundness", c: "#3f7fd0", m: [["S.1", "Connectivity", "rule-based geometry"], ["S.2", "Collision", "rule-based geometry"], ["S.3", "Stability", "rule-based geometry"]]},
  {key: "Affordance", c: "#159488", m: [["A.1", "Geometry", "against the reference"], ["A.2", "Parts", "Wikipedia-grounded"], ["A.3", "Kinematics", "Wikipedia-grounded"]]},
  {key: "Design", c: "#7a5bd6", m: [["D.1", "Decomposition", "calibrated VLM + rules"], ["D.2", "Aesthetics", "calibrated VLM"], ["D.3", "Alignment", "calibrated VLM + rules"]]},
  {key: "Realization", c: "#d0782f", m: [["R.1", "Sequence", "assembly order"], ["R.2", "Material", "Wikipedia-grounded"], ["R.3", "Operability", "simulation"]]}];
const METRICS = LEVELS.flatMap(l => l.m.map(x => x[0]));
$("#levels").innerHTML = LEVELS.map((l, i) => `<div class="card level" style="--c:${l.c}"><h4><small>Level ${i + 1}</small>${l.key}</h4>
  <ul>${l.m.map(([k, n, h]) => `<li><b>${k}</b>${n} <span>· ${h}</span></li>`).join("")}</ul></div>`).join("");

function heat(v) {
  if (v == null) return "";
  const lo = [158, 197, 232], mid = [236, 236, 236], hi = [245, 177, 131], t = Math.max(0, Math.min(1, v / 100));
  const [a, b, u] = t < .5 ? [lo, mid, t / .5] : [mid, hi, (t - .5) / .5];
  return `background:rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * u)).join(",")})`;
}
let RES = null, LB_G = "agents", LB_SORT = {k: "avg", dir: -1};
function drawLB() {
  if (!RES) return;
  const rows = RES.rows.map(r => ({...r, avg: r.scores.includes(null) ? null : r.scores.reduce((a, b) => a + b, 0) / 12, g: GROUP[r.id]}));
  const ranked = rows.filter(r => r.avg != null).sort((a, b) => b.avg - a.avg);
  const rank = Object.fromEntries(ranked.map((r, i) => [r.id, i + 1]));
  let xs = rows.filter(r => LB_G === "all" || (LB_G === "agents" ? r.g !== "ext" : r.g === LB_G));
  const val = r => LB_SORT.k === "avg" ? r.avg : r.scores[METRICS.indexOf(LB_SORT.k)];
  const ordered = LB_G === "all" && LB_SORT.k === "avg" && LB_SORT.dir < 0;
  if (ordered) xs = ["closed", "open", "ext"].flatMap(g => ORDER[g].map(id => xs.find(r => r.id === id)).filter(Boolean));
  else xs.sort((a, b) => {
    const va = val(a), vb = val(b);
    if (va == null || vb == null) return (va == null) - (vb == null);
    return LB_SORT.dir < 0 ? vb - va : va - vb;
  });
  const th = (k, label) => `<th class="sortable" data-k="${k}"${LB_SORT.k === k ? ` aria-sort="${LB_SORT.dir < 0 ? "descending" : "ascending"}"` : ""}>${label}<span class="arr">${LB_SORT.dir < 0 ? "▼" : "▲"}</span></th>`;
  let h = `<thead><tr class="lv"><th></th><th></th>${LEVELS.map(l => `<th colspan="3" style="--c:${l.c}"><span>${l.key}</span></th>`).join("")}<th></th></tr>
    <tr><th>#</th><th style="text-align:left;padding-left:.75rem">System</th>${LEVELS.flatMap(l => l.m.map(([k, n]) => th(k, `<span title="${n}">${k}</span>`))).join("")}${th("avg", "Mean")}</tr></thead><tbody>`;
  const GT = {closed: "Frontier closed-source APIs", open: "Open-source (multimodal) LLMs", ext: "Domain-specific generators"};
  let last = null;
  for (const r of xs) {
    if (ordered && r.g !== last) { h += `<tr class="grp"><td colspan="15">${GT[r.g]}</td></tr>`; last = r.g; }
    const k = rank[r.id];
    const rk = k ? (k <= 3 ? `<span class="medal m${k}">${k}</span>` : k) : "–";
    h += `<tr><td class="c-rank">${rk}</td><td class="c-model"><span class="mname">${who(r.id)}</span></td>` +
      r.scores.map(v => v == null ? `<td class="cell na">N/A</td>` : `<td class="cell" style="${heat(v)}">${v.toFixed(1)}</td>`).join("") +
      `<td class="c-avg">${r.avg == null ? '<span class="muted">–</span>' : `${r.avg.toFixed(1)}<span class="bar" style="width:${r.avg}%"></span>`}</td></tr>`;
  }
  $("#lb").innerHTML = h + "</tbody>";
  $$("#lb th.sortable").forEach(t => t.onclick = () => {
    LB_SORT = LB_SORT.k === t.dataset.k ? {k: t.dataset.k, dir: -LB_SORT.dir} : {k: t.dataset.k, dir: -1};
    drawLB();
  });
}
segment($("#lb-filter"), g => { LB_G = g; LB_SORT = {k: g === "ext" ? "S.1" : "avg", dir: -1}; drawLB(); });

// level means for closed vs open agents, the gap figure in Discussion
function drawGap() {
  const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
  const grp = g => RES.rows.filter(r => GROUP[r.id] === g);
  const lv = (rows, i) => mean(rows.map(r => mean(r.scores.slice(3 * i, 3 * i + 3))));
  const C = grp("closed"), O = grp("open");
  $("#gapfig").innerHTML = `<p class="gf-head">Mean score per level</p>` + LEVELS.map((l, i) => {
    const c = lv(C, i), o = lv(O, i);
    return `<div class="gf-row"><span class="gf-l">${l.key}</span><div class="gf-bars">
      <div class="gf-bar"><i style="width:${c}%;background:${l.c}"></i><b>${c.toFixed(1)}</b></div>
      <div class="gf-bar o"><i style="width:${o}%;background:${l.c}"></i><b>${o.toFixed(1)}</b></div></div></div>`;
  }).join("") + `<p class="gf-key"><span><i class="sw1"></i>Closed-source APIs (${C.length})</span><span><i class="sw2"></i>Open-source LLMs (${O.length})</span></p>`;
}

// ---- explorer: all 200 tasks, every model's output --------------------------
let XI = [], XSRC = "All", XALL = false, CUR = null, CMP_TIER = "B", CMP_ROUND = "r1";
const XCACHE = {};
function drawThumbs() {
  const xs = XI.filter(e => XSRC === "All" || e.source === XSRC), ys = XALL ? xs : xs.slice(0, 36);
  $("#thumbs").innerHTML = ys.map(e => `<button type="button" class="thumb${e.ready ? "" : " pending"}" data-t="${e.task}">
    <img src="${e.thumb}" alt="" loading="lazy"><span class="tt"><span class="tn">${esc(cap(e.name))}</span>
    <span class="ts">${esc(e.source)}${e.ready ? ` · ${e.n_out} outputs` : " · rendering"}</span></span></button>`).join("");
  $$("#thumbs .thumb").forEach(b => b.onclick = () => openTask(b.dataset.t));
  const more = $("#thumbs-more");
  more.hidden = XALL || xs.length <= 36; more.textContent = `Show all ${xs.length}`;
}
async function openTask(task, keepScroll) {
  const d = XCACHE[task] || (XCACHE[task] = await getJSON(`data/explore/${task}.json`));
  if (!d) return;
  CUR = d;
  if (!d.C.length && CMP_TIER === "C") CMP_TIER = "B";
  $("#stage").hidden = false;
  drawStage();
  if (!keepScroll) $("#stage").scrollIntoView({behavior: "smooth", block: "start"});
  history.replaceState(null, "", "#task=" + task);
}
function setSeg(el, v) { $$("button", el).forEach(b => b.setAttribute("aria-pressed", b.dataset.v === v)); }
function drawStage() {
  const o = CUR;
  $("#st-title").textContent = cap(o.name);
  const xs = listed(), i = xs.findIndex(e => e.task === o.task);
  $("#st-pos").textContent = i >= 0 ? `${i + 1} / ${xs.length}` : "";
  const tc = $('#cmp-tier [data-v="C"]'), r3 = $('#cmp-round [data-v="r3"]');
  tc.disabled = !o.C.length; tc.style.opacity = o.C.length ? "" : ".4"; tc.title = o.C.length ? "" : "Create-only runs exist for the original 50 tasks";
  r3.disabled = !o.has_r3; r3.style.opacity = o.has_r3 ? "" : ".4"; r3.title = o.has_r3 ? "" : "Round-3 renders are shown for the ten curated objects";
  if (!o.C.length) CMP_TIER = "B";
  if (!o.has_r3) CMP_ROUND = "r1";
  setSeg($("#cmp-tier"), CMP_TIER); setSeg($("#cmp-round"), CMP_ROUND);
  $("#inputcard").innerHTML = `<span class="lab">Input</span>
    <img src="${o.input}" alt="Input image for ${esc(o.name)}" data-zoom="${o.input}" data-zcap="Input image ${esc(o.image_id)}">
    <p class="prompt"><span class="pl">Instruction</span>${esc(o.prompt)}</p>
    <span class="lab2">Reference (not shown to the agent)</span>
    <img class="refimg" src="${o.reference}" alt="Reference structure, assembled and exploded" data-zoom="${o.reference}" data-zcap="Reference: assembled and exploded">
    <p class="facts">${esc(o.source)} · ${o.parts} reference parts${o.difficulty ? ` · ${o.difficulty}` : ""}. Parts are coloured by the role each agent gave them.</p>`;
  const rows = o[CMP_TIER] || [];
  const card = r => {
    const im = r[CMP_ROUND] || (CMP_ROUND === "r3" ? null : r.r1);
    const stat = im ? `${im.n_parts} part${im.n_parts === 1 ? "" : "s"}${im.n_parts ? ` · ${im.n_created} created` : ""}` : "";
    const shot = im && im.img ? `<div class="shot" data-zoom="${im.img}" data-zcap="${esc(NAME[r.id] || r.id)} · ${esc(o.name)}"><img src="${im.img}" alt="${esc(o.name)} by ${esc(NAME[r.id] || r.id)}" loading="lazy"></div>`
      : `<div class="none">${im && im.n_parts === 0 ? "empty design" : "no design submitted"}</div>`;
    return `<article class="card mcard">${shot}<div class="meta"><div class="who" title="${esc(NAME[r.id] || r.id)}">${who(r.id)}</div><div class="stat">${stat || "&nbsp;"}</div></div></article>`;
  };
  const sec = (g, label) => { const xs = ORDER[g].map(id => rows.find(r => r.id === id)).filter(Boolean); return xs.length ? `<p class="grp-title">${label}</p>` + xs.map(card).join("") : ""; };
  let h = sec("closed", "Frontier closed-source APIs") + sec("open", "Open-source LLMs");
  if (CMP_TIER === "B" && CMP_ROUND === "r1" && o.EXT.length) {
    const xs = ORDER.ext.map(id => o.EXT.find(r => r.id === id)).filter(Boolean);
    h += `<p class="grp-title">Domain-specific generators (one shot)</p>` + xs.map(r => {
      const shot = r.img ? `<div class="shot" data-zoom="${r.img}" data-zcap="${esc(NAME[r.id])} · ${esc(o.name)}"><img src="${r.img}" alt="${esc(o.name)} by ${esc(NAME[r.id])}" loading="lazy"></div>` : `<div class="none">no output</div>`;
      return `<article class="card mcard">${shot}<div class="meta"><div class="who" title="${esc(NAME[r.id] || r.id)}">${who(r.id)}</div><div class="stat">${r.n_parts} part${r.n_parts === 1 ? "" : "s"} · ${r.cond === "image" ? "image" : "name"} input</div></div></article>`;
    }).join("");
  }
  $("#mgrid").className = "mgrid dense";
  $("#mgrid").innerHTML = h || `<p class="note">Outputs for this task are still being rendered.</p>`;
  $("#cmp-note").innerHTML = CMP_ROUND === "r1"
    ? `<b>Round 1</b>: the design each agent first submitted, the one scored in the leaderboard. Main setting: object name + image.`
    : `<b>Round 3</b>: the design after two more rounds of inspecting renders and error reports. Revisions add about 2 points on average.`;
}
segment($("#cmp-tier"), v => { CMP_TIER = v; drawStage(); });
segment($("#cmp-round"), v => { CMP_ROUND = v; drawStage(); });
const listed = () => XI.filter(e => XSRC === "All" || e.source === XSRC);
function step(d) {
  const xs = listed(), i = xs.findIndex(e => e.task === CUR?.task);
  if (i < 0 || !xs.length) return;
  openTask(xs[(i + d + xs.length) % xs.length].task, true);
}
$("#st-prev").onclick = () => step(-1);
$("#st-next").onclick = () => step(1);
document.addEventListener("keydown", e => {
  if ($("#stage").hidden || !lb.hidden || /input|select|textarea/i.test(e.target.tagName)) return;
  if (e.key === "ArrowLeft") step(-1);
  else if (e.key === "ArrowRight") step(1);
  else if (e.key === "Escape") $("#st-close").click();
});
$("#st-close").onclick = () => { $("#stage").hidden = true; history.replaceState(null, "", "#compare"); $("#srcchips").scrollIntoView({behavior: "smooth", block: "center"}); };
$("#thumbs-more").onclick = () => { XALL = true; drawThumbs(); };

// ---- step / function galleries -----------------------------------------------
let MEDIA = null;
function drawSteps(P) {
  const xs = byGroup((MEDIA.step || []).filter(m => P === "all" || m.protocol === P));
  $("#steps").innerHTML = xs.map(m => ioCard(m, {extra: m.created_share != null ? `<br><span class="muted">${share(m)}</span>` : ""})).join("");
}
const FUNC = {
  joints: {label: "Joints", lede: "Each agent declares joints between its parts. Shown: one declared joint swept through its range, with the moving parts in their declared motion."},
  materials: {label: "Materials", lede: "Each part is assigned a material. Shown: the build coloured by the material the agent declared for each part.", cls: "mat"},
  sequence: {label: "Assembly sequences", lede: "Each agent declares the order in which its parts are put together. Shown: the declared assembly sequence, played back step by step."}};
function drawFunc(k) {
  const f = FUNC[k];
  $("#func-lede").textContent = f.lede;
  $("#func").innerHTML = byGroup(MEDIA[k] || []).map(m => ioCard(m, {imgClass: f.cls || "",
    extra: m.materials ? `<br><span class="muted">${m.materials.slice(0, 5).map(esc).join(", ")}${m.materials.length > 5 ? "…" : ""}</span>` : m.n_steps ? `<br><span class="muted">${m.n_steps} assembly steps</span>` : ""})).join("");
  $("#jointfigs-wrap").hidden = k !== "joints";
}
function chips(el, items, cb, first) {
  el.innerHTML = items.map(([v, label, n]) => `<button type="button" class="chip" data-v="${esc(v)}" aria-pressed="${v === first}">${label}${n != null ? ` <span class="n">${n}</span>` : ""}</button>`).join("");
  $$(".chip", el).forEach(b => b.onclick = () => { $$(".chip", el).forEach(x => x.setAttribute("aria-pressed", x === b)); cb(b.dataset.v); });
}

// ---- dataset table -----------------------------------------------------------
const T1 = [["LEGO toy data", "BrickNet", "320,808", "5", "13.80 / 14.60", "888", "8.79 / 10.82"], ["", "BrickComposer", "9,977", "50", "10.16 / 10.34", "50", "10.16 / 10.34"],
  ["Static structures", "PartNeXt", "23,519", "14", "5.07 / 8.00", "589", "6.00 / 7.92"], ["Mechanical structures", "Fusion 360 Gallery (Joinable)", "8,251", "55", "6.85 / 8.91", "80", "6.20 / 8.41"],
  ["Daily objects / furniture", "Artiverse", "5,402", "42", "6.50 / 8.57", "908", "5.90 / 7.43"], ["Product-level CAD", "Robotics & vehicles", "7", "7", "8.29 / 9.29", "7", "8.29 / 9.29"],
  ["(open-source hardware)", "Consumer devices", "10", "10", "6.40 / 7.80", "10", "6.40 / 7.80"], ["", "Lab & medical instruments", "7", "7", "5.71 / 8.43", "7", "5.71 / 8.43"],
  ["", "Fabrication machines & tools", "10", "10", "6.30 / 9.60", "10", "6.30 / 9.60"], ["Total in LMBuild", "", "367,991", "200", "7.62 / 9.25", "2,549", "7.03 / 8.83"]];
$("#t1").innerHTML = `<thead><tr><th rowspan="2" class="l">Geometry type</th><th rowspan="2" class="l">Data source</th><th rowspan="2">Original size</th><th colspan="2">LMBuild-Core</th><th colspan="2">LMBuild-Full</th></tr>
  <tr><th>Size</th><th>Avg. parts / aff.</th><th>Size</th><th>Avg. parts / aff.</th></tr></thead><tbody>` +
  T1.map((r, i) => `<tr${i === T1.length - 1 ? ' class="tot"' : ""}><td class="l">${r[0]}</td><td class="l">${r[1]}</td>${r.slice(2).map(x => `<td>${x}</td>`).join("")}</tr>`).join("") + "</tbody>";

// ---- load everything -----------------------------------------------------------
Promise.all(["results", "captions", "objects", "media", "inputs", "explore/index"].map(n => getJSON(`data/${n}.json`))).then(([R, C, O, M, I, X]) => {
  INPUTS = I || {};
  if (C) $$("[data-cap]").forEach(e => { const k = e.dataset.cap; if (C[k]) e.innerHTML = `<b>${k}.</b> ` + esc(C[k].replace(/^(Figure|Table) \d+:\s*/, "")); });
  if (R) { RES = R; drawLB(); drawGap(); }

  if (M) {
    MEDIA = M;
    const teaser = (M.step || []).filter(m => m.teaser).slice(0, 6);
    $("#teaser").innerHTML = teaser.map(m => ioCard(m)).join("");
    const PR = ["all", "baseline", "retrieval-only", "creation-only"];
    chips($("#stepf"), PR.map(p => [p, p === "all" ? "All" : cap(PLAB[p]), (M.step || []).filter(m => p === "all" || m.protocol === p).length]), drawSteps, "all");
    drawSteps("all");
    chips($("#funcf"), Object.entries(FUNC).map(([k, f]) => [k, f.label, (M[k] || []).length]), drawFunc, "joints");
    drawFunc("joints");
    $("#jointfigs").innerHTML = [5, 6, 7, 8, 9, 10, 11, 12].map(n => `<figure class="card figure"><img src="assets/paper/fig${n}.webp" loading="lazy" alt="Figure ${n}" data-zoom="assets/paper/fig${n}.webp"><p class="cap">${C && C["Figure " + n] ? `<b>Figure ${n}.</b> ` + esc(C["Figure " + n].replace(/^Figure \d+:\s*/, "")) : ""}</p></figure>`).join("");
  }

  if (X) {
    XI = X;
    const SRC = ["All", "Product CAD", "Fusion 360", "Artiverse", "PartNeXt", "BrickComposer", "BrickNet"];
    chips($("#srcchips"), SRC.map(s => [s, s, X.filter(e => s === "All" || e.source === s).length]), v => { XSRC = v; XALL = false; drawThumbs(); }, "All");
    drawThumbs();
    const m = location.hash.match(/^#task=(.+)$/);
    if (m) { showTab("compare"); openTask(decodeURIComponent(m[1])); }
  }
  if (O) {
    const SRC = ["All", "Artiverse", "Fusion 360", "BrickComposer", "Product CAD", "PartNeXt", "BrickNet"];
    let S = "All", ALL = false;
    const draw = () => {
      const xs = O.filter(o => S === "All" || o.source === S), ys = ALL ? xs : xs.slice(0, 24);
      $("#gal").innerHTML = ys.map(o => `<article class="card io"><div class="io-out"><img src="${o.img}" loading="lazy" alt="${esc(o.name)}" data-zoom="${o.img}" data-zcap="${esc(o.name)} · ${esc(o.source)}"></div>
        <div class="io-cap"><span class="who">${esc(cap(o.name))}</span>${esc(o.source)} · ${o.parts} parts</div></article>`).join("");
      $("#more").hidden = ALL || xs.length <= 24;
      $("#more").textContent = `Show all ${xs.length}`;
    };
    chips($("#srcf"), SRC.map(s => [s, s, O.filter(o => s === "All" || o.source === s).length]), s => { S = s; ALL = false; draw(); }, "All");
    $("#more").onclick = () => { ALL = true; draw(); };
    draw();
  }
});
