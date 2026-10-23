/* Alexandria dashboard: renders data.json (built by `alexandria dashboard build`). Panel types: see PANELS.md. */
"use strict";

const S = { data: null, tab: null, charts: [], built: null, slots: new Map() };

/* Series colours: 8 fixed slots (CSS --series-1..8), never cycled. A series NAME keeps one slot on the whole page:
   pinned names first, then first-seen order over the page's panels; names beyond 8 slots fold into "Other". */
const N_SLOTS = 8;
const PINNED = { "arxiv/new": 1, "arxiv/old": 2, "arxiv/titled": 3, "pubmed": 4, "bethgelab": 5, "web-papers": 7 };
function prepareSlots(panels) {
  const names = [];
  panels.forEach(p => { if (p.type === "bar" || p.type === "line") (p.series || []).forEach(s => { if (!names.includes(s.name)) names.push(s.name); }); });
  S.slots = new Map();
  names.forEach(n => { if (PINNED[n]) S.slots.set(n, PINNED[n]); });
  const used = new Set(S.slots.values());
  names.forEach(n => {
    if (S.slots.has(n)) return;
    let k = 1; while (k <= N_SLOTS && used.has(k)) k++;
    if (k <= N_SLOTS) { S.slots.set(n, k); used.add(k); }
  });
}
function seriesColor(name) { const k = S.slots.get(name); return cssVar(k ? `--series-${k}` : "--series-other"); }
function foldSeries(series) {
  /* > N_SLOTS series in one chart, or series without a slot: sum them into one "Other" series */
  const keep = [], rest = [];
  series.forEach(s => (S.slots.has(s.name) && keep.length < N_SLOTS - 1 ? keep : rest).push(s));
  if (rest.length <= 1 && keep.length + rest.length <= N_SLOTS) return series;
  const n = Math.max(...rest.map(s => (s.data || []).length));
  const other = { name: "Other", data: Array.from({ length: n }, (_, i) => rest.reduce((a, s) => a + (isNum(s.data[i]) ? s.data[i] : 0), 0)) };
  return [...keep, other];
}
function pagePanels() {
  if (!S.data || S.tab === "Overview" || S.tab === "Runs") return [];
  return (S.data.workstreams[S.tab.toLowerCase()] || []).flatMap(st => st.panels || []);
}

/* ------------------------------------------------------------------ small utils */
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } return null; }
function esc(s) { return String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function el(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") e.className = v; else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false) e.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return e;
}
function cssVar(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }
function isNum(v) { return typeof v === "number" && isFinite(v); }
function fmt(v, f) {
  if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "–";
  if (Array.isArray(v)) return v.join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  if (typeof v !== "number") { if (typeof v === "boolean") return v ? "yes" : "no"; return String(v); }
  switch (f) {
    case "int": return Math.round(v).toLocaleString("en-US");
    case "pct": return (v * 100).toFixed(1) + "%";
    case "pct1": return (v * 100).toFixed(1) + "%";
    case "float": return v.toLocaleString("en-US", { maximumFractionDigits: 2 });
    case "float3": return v.toFixed(3);
    default: return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toLocaleString("en-US", { maximumFractionDigits: 3 });
  }
}
function stCls(s) { return "st-" + String(s || "not run").replace(/\s+/g, ""); }
function badge(s) { return el("span", { class: "badge " + stCls(s) }, s || "not run"); }
function hhmm(iso) { if (!iso) return "–"; const d = new Date(iso); return isNaN(d) ? iso : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
function dur(s) { if (!isNum(s)) return "–"; if (s < 90) return s.toFixed(0) + " s"; if (s < 5400) return (s / 60).toFixed(1) + " min"; return (s / 3600).toFixed(2) + " h"; }

/* ------------------------------------------------------------------ theme */
function applyTheme() {
  const t = store("alex-theme") || "auto";
  if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  document.getElementById("theme").textContent = "Theme: " + t;
}
document.getElementById("theme").addEventListener("click", () => {
  const order = ["auto", "light", "dark"], t = store("alex-theme") || "auto";
  store("alex-theme", order[(order.indexOf(t) + 1) % 3]); applyTheme(); render();
});
applyTheme();

/* ------------------------------------------------------------------ Chart.js: whiskers for lo/hi */
const whiskers = {
  id: "whiskers",
  afterDatasetsDraw(chart) {
    const ctx = chart.ctx, horiz = chart.options.indexAxis === "y";
    chart.data.datasets.forEach((ds, di) => {
      if (!ds._lo || !ds._hi) return;
      const meta = chart.getDatasetMeta(di); if (meta.hidden) return;
      const vs = horiz ? chart.scales.x : chart.scales.y;
      ctx.save(); ctx.strokeStyle = cssVar("--text"); ctx.lineWidth = 1.2;
      meta.data.forEach((bar, i) => {
        const lo = ds._lo[i], hi = ds._hi[i]; if (!isNum(lo) || !isNum(hi)) return;
        const a = vs.getPixelForValue(lo), b = vs.getPixelForValue(hi), c = horiz ? bar.y : bar.x, w = 4;
        ctx.beginPath();
        if (horiz) { ctx.moveTo(a, c); ctx.lineTo(b, c); ctx.moveTo(a, c - w); ctx.lineTo(a, c + w); ctx.moveTo(b, c - w); ctx.lineTo(b, c + w); }
        else { ctx.moveTo(c, a); ctx.lineTo(c, b); ctx.moveTo(c - w, a); ctx.lineTo(c + w, a); ctx.moveTo(c - w, b); ctx.lineTo(c + w, b); }
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};

function chartTable(p, series) {
  /* the same numbers as the chart, for screen readers and for colours with low contrast */
  const cols = [{ key: "_label", label: p.horizontal ? "category" : "x" }];
  series.forEach((s, i) => {
    cols.push({ key: "s" + i, label: s.name || "value", fmt: p.fmt });
    if (s.lo && s.hi) cols.push({ key: "ci" + i, label: (s.name ? s.name + " " : "") + "95% CI" });
  });
  const rows = (p.labels || []).map((l, j) => {
    const r = { _label: l };
    series.forEach((s, i) => { r["s" + i] = s.data[j]; if (s.lo && s.hi) r["ci" + i] = `${fmt(s.lo[j], p.fmt)} – ${fmt(s.hi[j], p.fmt)}`; });
    return r;
  });
  return tablePanel({ columns: cols, rows, limit: 100, bare: true });
}

function chartPanel(p, kind) {
  const toggle = el("button", { type: "button", class: "more", "aria-pressed": "false" }, "table");
  const card = el("div", { class: "card" + (p.wide ? " wide" : "") },
    el("div", { class: "cardhead" }, el("div", {}, p.title ? el("h3", {}, p.title) : null, p.hint ? el("p", { class: "hint" }, p.hint) : null), toggle));
  const labels = p.labels || [], horiz = !!p.horizontal, many = labels.length > 14;
  const series = foldSeries(p.series || []);
  const box = el("div", { class: "chart" + ((horiz && labels.length > 10) || many ? " tall" : "") });
  const tbl = chartTable(p, series); tbl.hidden = true;
  toggle.addEventListener("click", () => { const show = tbl.hidden; tbl.hidden = !show; box.hidden = show; toggle.setAttribute("aria-pressed", String(show)); toggle.textContent = show ? "chart" : "table"; });
  card.append(box, tbl);
  if (!window.Chart) { box.append(el("p", { class: "empty" }, "Chart.js did not load (offline and vendor file missing): use the table view.")); return card; }
  const canvas = el("canvas", { role: "img", "aria-label": `${p.title || kind} chart; press the table button for the values` });
  box.append(canvas);
  const text = cssVar("--text"), muted = cssVar("--muted"), grid = cssVar("--grid"), surface = cssVar("--panel"), f = p.fmt;
  const datasets = series.map(s => {
    const col = seriesColor(s.name);
    const d = kind === "line"
      ? { label: s.name, data: s.data, borderColor: col, backgroundColor: col, borderWidth: 2, pointRadius: 4, pointHoverRadius: 6, tension: 0.15 }
      : { label: s.name, data: s.data, backgroundColor: col, borderColor: surface, borderWidth: 2, borderRadius: 4, borderSkipped: "start",
          maxBarThickness: 22, categoryPercentage: 0.75, barPercentage: 0.9 };
    if (s.lo && s.hi) { d._lo = s.lo; d._hi = s.hi; }
    return d;
  });
  const valueAxis = { stacked: !!p.stacked, beginAtZero: kind === "bar", grid: { color: grid }, border: { color: grid }, ticks: { color: muted, callback: v => fmt(v, f) } };
  const catAxis = { stacked: !!p.stacked, grid: { display: false }, border: { color: grid }, ticks: { color: muted, autoSkip: !horiz, maxRotation: 60, callback(v) { const s = String(this.getLabelForValue(v)); return s.length > 28 ? s.slice(0, 27) + "…" : s; } } };
  const chart = new Chart(canvas, {
    type: kind, data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, indexAxis: horiz ? "y" : "x",
      interaction: kind === "line" ? { mode: "index", intersect: false } : { mode: "nearest", intersect: true },
      scales: horiz ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
      plugins: {
        legend: { display: datasets.length >= 2, labels: { color: text, usePointStyle: true, pointStyle: kind === "line" ? "circle" : "rectRounded" } },
        tooltip: { titleColor: text, bodyColor: text, backgroundColor: surface, borderColor: grid, borderWidth: 1,
          callbacks: { label(c) { const ds = c.dataset; let s = (ds.label ? ds.label + ": " : "") + fmt(c.parsed[horiz ? "x" : "y"], f); if (ds._lo) s += `  [${fmt(ds._lo[c.dataIndex], f)}, ${fmt(ds._hi[c.dataIndex], f)}]`; return s; } } },
      },
    },
    plugins: [whiskers],
  });
  S.charts.push(chart);
  return card;
}

/* ------------------------------------------------------------------ panel renderers */
function kpisPanel(p) {
  return el("div", { class: "kpis wide" }, (p.items || []).map(it =>
    el("div", { class: "kpi " + (it.status || "") }, el("div", { class: "v" }, fmt(it.value, it.fmt)), el("div", { class: "l" }, it.label),
      it.status ? el("div", { class: "h" }, it.status === "good" ? "OK" : it.status === "warn" ? "check" : "problem") : null,
      it.hint ? el("div", { class: "h" }, it.hint) : null)));
}

function tablePanel(p) {
  const cols = p.columns || (p.rows && p.rows.length ? Object.keys(p.rows[0]).map(k => ({ key: k, label: k })) : []);
  const card = el("div", { class: "card" + (cols.length > 5 || p.wide ? " wide" : "") }, p.title ? el("h3", {}, p.title) : null, p.hint ? el("p", { class: "hint" }, p.hint) : null);
  let rows = (p.rows || []).slice(), sortKey = p.sort || null, dir = p.sort ? -1 : 1, showAll = false;
  const limit = p.limit || 50, wrap = el("div", { class: "tablewrap" });
  const isNumCol = c => rows.some(r => isNum(r[c.key])) && !rows.some(r => r[c.key] != null && !isNum(r[c.key]));
  function draw() {
    if (sortKey) rows.sort((a, b) => { const x = a[sortKey], y = b[sortKey]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    const shown = showAll ? rows : rows.slice(0, limit);
    const head = el("tr", {}, cols.map(c => el("th", { scope: "col", tabindex: "0", "aria-sort": sortKey === c.key ? (dir > 0 ? "ascending" : "descending") : null,
      onclick: () => { if (sortKey === c.key) dir = -dir; else { sortKey = c.key; dir = isNumCol(c) ? -1 : 1; } draw(); },
      onkeydown: e => { if (e.key === "Enter") e.target.click(); } }, c.label || c.key)));
    const cell = (r, c) => {
      if (c.fmt !== "md") return el("td", { class: isNum(r[c.key]) ? "num" : null }, fmt(r[c.key], c.fmt));
      const td = el("td", {}); td.innerHTML = mdInline(esc(r[c.key] ?? "")); return td;
    };
    const body = shown.map(r => el("tr", {}, cols.map(c => cell(r, c))));
    wrap.replaceChildren(el("table", {}, el("thead", {}, head), el("tbody", {}, body)));
    if (rows.length > limit) wrap.append(el("button", { type: "button", class: "more", onclick: () => { showAll = !showAll; draw(); } }, showAll ? "show fewer" : `show all ${rows.length} rows`));
    if (!rows.length) wrap.append(el("p", { class: "empty" }, "no rows"));
  }
  draw();
  if (p.bare) return wrap;
  card.append(wrap); return card;
}

function clampText(text) {
  const t = String(text ?? ""), box = el("div", { class: "clamp" }, t);
  if (t.length < 500) { box.classList.add("open"); return [box]; }
  const btn = el("button", { type: "button", class: "more", onclick: () => { box.classList.toggle("open"); btn.textContent = box.classList.contains("open") ? "less" : "more"; } }, "more");
  return [box, btn];
}

function examplesPanel(p) {
  const card = el("div", { class: "card wide" }, el("h3", {}, p.title || "Examples"), p.hint ? el("p", { class: "hint" }, p.hint) : null);
  const items = p.items || [];
  if (!items.length) card.append(el("p", { class: "empty" }, "none"));
  items.forEach((it, i) => card.append(el("details", { class: "ex", open: i === 0 && items.length < 4 },
    el("summary", {}, it.title || `example ${i + 1}`, (it.tags || []).map(t => el("span", { class: "tag" }, t))),
    Object.entries(it.fields || {}).map(([k, v]) => el("div", { class: "field" }, el("div", { class: "k" }, k), ...clampText(typeof v === "string" ? v : JSON.stringify(v, null, 1)))))));
  return card;
}

function mdInline(s) {
  return s.replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>")
    .replace(/\[([^\]]+)\]\(((?:https?:\/\/|\.{0,2}\/|#)[^)\s]*)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
}
function markdown(text) {
  const lines = esc(text || "").split("\n"); let html = "", list = null, code = false, para = [];
  const flushP = () => { if (para.length) { html += "<p>" + mdInline(para.join(" ")) + "</p>"; para = []; } };
  const flushL = () => { if (list) { html += `</${list}>`; list = null; } };
  for (const ln of lines) {
    if (ln.trim().startsWith("```")) { flushP(); flushL(); html += code ? "</pre>" : "<pre>"; code = !code; continue; }
    if (code) { html += ln + "\n"; continue; }
    let m;
    if ((m = ln.match(/^(#{1,3})\s+(.*)/))) { flushP(); flushL(); html += `<h${m[1].length + 1}>${mdInline(m[2])}</h${m[1].length + 1}>`; }
    else if ((m = ln.match(/^\s*[-*]\s+(.*)/)) || (m = ln.match(/^\s*\d+[.)]\s+(.*)/))) {
      flushP(); const t = /^\s*\d/.test(ln) ? "ol" : "ul"; if (list !== t) { flushL(); html += `<${t}>`; list = t; } html += "<li>" + mdInline(m[1]) + "</li>";
    } else if (!ln.trim()) { flushP(); flushL(); }
    else { flushL(); para.push(ln.trim()); }
  }
  flushP(); flushL(); if (code) html += "</pre>";
  return html;
}
function markdownPanel(p) { return el("div", { class: "card wide md", html: markdown(p.text) }); }

function imagePanel(p) {
  const card = el("div", { class: "card" + (p.wide ? " wide" : "") }, p.title ? el("h3", {}, p.title) : null);
  card.append(p.missing ? el("p", { class: "empty" }, `image missing: ${p.src}`) : el("img", { class: "panel", src: p.src, alt: p.alt || p.title || "figure", loading: "lazy" }));
  return card;
}

function panel(p) {
  switch (p.type) {
    case "kpis": return kpisPanel(p);
    case "table": return tablePanel(p);
    case "bar": return chartPanel(p, "bar");
    case "line": return chartPanel(p, "line");
    case "examples": return examplesPanel(p);
    case "markdown": return markdownPanel(p);
    case "image": return imagePanel(p);
    default: return el("div", { class: "card" }, el("p", { class: "empty" }, `unknown panel type: ${p.type}`));
  }
}

/* ------------------------------------------------------------------ pages */
function stageSection(st) {
  const sec = el("section", { class: "section", id: `s-${st.ws}-${st.stage}` },
    el("div", { class: "head" }, el("h2", {}, st.title || st.stage), badge(st.status), el("span", { class: "muted" }, `${st.ws}/${st.stage}` + (st.updated ? " · updated " + hhmm(st.updated) : ""))));
  if (st.summary) sec.append(el("div", { class: "summary md", html: markdown(st.summary) }));
  const grid = el("div", { class: "grid" });
  (st.panels || []).forEach(p => {
    const node = panel(p);
    if (p.type === "kpis") { node.style.gridColumn = "1/-1"; }
    grid.append(node);
  });
  if (!(st.panels || []).length) grid.append(el("p", { class: "empty" }, st.status === "not run" ? "not run yet" : "no panels published by this stage"));
  sec.append(grid); return sec;
}

function wsPage(ws) {
  const stages = S.data.workstreams[ws] || [];
  const jump = el("div", { class: "toolbar" }, el("span", { class: "muted" }, "Jump to:"),
    stages.map(st => el("button", { type: "button", onclick: () => document.getElementById(`s-${st.ws}-${st.stage}`).scrollIntoView({ behavior: "smooth" }) }, st.stage, " ", el("span", { class: "badge " + stCls(st.status) }, st.status))));
  return [jump, ...stages.map(stageSection)];
}

function dag(ws, reg) {
  const nodes = reg.filter(r => r.ws === ws), byName = Object.fromEntries(nodes.map(n => [n.name, n])), depth = {};
  const d = (n, seen = new Set()) => { if (depth[n.name] != null) return depth[n.name]; if (seen.has(n.name)) return 0; seen.add(n.name);
    const deps = (n.needs || []).map(x => byName[x]).filter(Boolean); return (depth[n.name] = deps.length ? 1 + Math.max(...deps.map(x => d(x, seen))) : 0); };
  nodes.forEach(n => d(n));
  const layers = []; nodes.forEach(n => (layers[depth[n.name]] ||= []).push(n));
  const wrap = el("div", { class: "dag" }), svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "edges"); svg.setAttribute("aria-hidden", "true"); wrap.append(svg);
  layers.forEach((ls, i) => wrap.append(el("div", { class: "layer" }, el("div", { class: "lt" }, i === 0 ? "inputs" : `step ${i}`),
    (ls || []).map(n => el("div", { class: "node " + String(n.status).replace(/\s+/g, ""), "data-name": n.name, title: n.help || "" },
      el("b", {}, n.name), " ", el("span", { class: "badge " + stCls(n.status) }, n.status),
      el("div", { class: "meta" }, `${n.kind} · ${n.resources} · env ${n.env}`), n.help ? el("div", { class: "meta" }, n.help) : null)))));
  const draw = () => {
    const box = wrap.getBoundingClientRect(); svg.replaceChildren();
    svg.setAttribute("width", wrap.scrollWidth); svg.setAttribute("height", wrap.scrollHeight);
    nodes.forEach(n => (n.needs || []).forEach(dep => {
      const a = wrap.querySelector(`[data-name="${CSS.escape(dep)}"]`), b = wrap.querySelector(`[data-name="${CSS.escape(n.name)}"]`); if (!a || !b) return;
      const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      const x1 = ra.right - box.left + wrap.scrollLeft, y1 = ra.top + ra.height / 2 - box.top, x2 = rb.left - box.left + wrap.scrollLeft, y2 = rb.top + rb.height / 2 - box.top;
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path"), mx = (x1 + x2) / 2;
      p.setAttribute("d", `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2 - 3},${y2} l-5,-4 m5,4 l-5,4`); svg.append(p);
    }));
  };
  requestAnimationFrame(draw); window.addEventListener("resize", draw, { passive: true });
  return wrap;
}

function runsTable(runs, limit) {
  return tablePanel({ title: limit ? "Latest runs" : null, limit: limit || 200, sort: "started",
    columns: [{ key: "started", label: "started" }, { key: "ws", label: "ws" }, { key: "stage", label: "stage" }, { key: "status", label: "status" },
      { key: "shardTxt", label: "task" }, { key: "dur", label: "time" }, { key: "host", label: "host" }, { key: "slurm_job", label: "job" }, { key: "error", label: "error" }],
    rows: runs.map(r => ({ ...r, started: hhmm(r.started), shardTxt: r.shard ? `${r.shard[0] + 1}/${r.shard[1]}` : "", dur: dur(r.seconds) })) });
}

function overviewPage() {
  const reg = S.data.registry || [], count = s => reg.filter(r => r.status === s).length;
  const out = [kpisPanel({ items: [
    { label: "stages registered", value: reg.length, fmt: "int" },
    { label: "done", value: count("done"), fmt: "int", status: count("done") ? "good" : null },
    { label: "running / partial", value: count("running") + count("partial"), fmt: "int", status: count("running") + count("partial") ? "warn" : null },
    { label: "failed", value: count("failed"), fmt: "int", status: count("failed") ? "bad" : "good" },
    { label: "skipped", value: count("skipped"), fmt: "int" },
    { label: "not run", value: count("not run"), fmt: "int" },
  ] })];
  [...new Set(reg.map(r => r.ws))].forEach(ws => out.push(el("div", { class: "card" }, el("h3", {}, `${ws.toUpperCase()} pipeline`),
    el("p", { class: "hint" }, "Each box is a stage you can run on its own (alexandria " + ws + " <stage>). Arrows: which stage's output it reads."), dag(ws, reg))));
  const runs = S.data.runs || [];
  out.push(runs.length ? runsTable(runs.slice(0, 15), 15) : el("p", { class: "empty" }, "no runs yet"));
  out.push(el("p", { class: "muted" }, "Results folder: ", el("code", {}, S.data.out_root || "")));
  return out;
}

function runsPage() {
  const runs = S.data.runs || [], wrap = el("div", {});
  const q = el("input", { type: "search", placeholder: "filter (stage, host, error…)", "aria-label": "filter runs", value: store("alex-runq") || "" });
  const st = el("select", { "aria-label": "status" }, ["all", "done", "failed", "running", "skipped"].map(s => el("option", { value: s }, s)));
  const body = el("div", {});
  const draw = () => {
    store("alex-runq", q.value); const t = q.value.toLowerCase();
    const sel = runs.filter(r => (st.value === "all" || r.status === st.value) && (!t || JSON.stringify([r.ws, r.stage, r.host, r.error, r.slurm_job]).toLowerCase().includes(t)));
    body.replaceChildren(runsTable(sel));
    const failed = sel.filter(r => r.traceback);
    if (failed.length) body.append(el("div", { class: "card wide" }, el("h3", {}, "Errors"), failed.map(r =>
      el("details", { class: "ex" }, el("summary", {}, `${r.ws}/${r.stage}`, el("span", { class: "tag" }, hhmm(r.started)), " ", el("span", { class: "err" }, r.error || "")), el("pre", {}, r.traceback)))));
  };
  q.addEventListener("input", draw); st.addEventListener("change", draw);
  wrap.append(el("div", { class: "toolbar" }, q, st), body); draw();
  return [wrap];
}

/* ------------------------------------------------------------------ shell */
function tabs() { return ["Overview", ...Object.keys(S.data.workstreams || {}).map(w => w.toUpperCase()), "Runs"]; }

function render() {
  if (!S.data) return;
  S.charts.forEach(c => c.destroy()); S.charts = [];
  const all = tabs(); if (!all.includes(S.tab)) S.tab = all.includes(store("alex-tab")) ? store("alex-tab") : "Overview";
  const nav = document.getElementById("tabs");
  nav.replaceChildren(...all.map(t => el("button", { type: "button", class: t === S.tab ? "on" : null, "aria-current": t === S.tab ? "page" : null,
    onclick: () => { S.tab = t; store("alex-tab", t); render(); window.scrollTo(0, 0); } }, t)));
  const main = document.getElementById("main"), y = window.scrollY;
  prepareSlots(pagePanels());
  const content = S.tab === "Overview" ? overviewPage() : S.tab === "Runs" ? runsPage() : wsPage(S.tab.toLowerCase());
  main.replaceChildren(...content);
  window.scrollTo(0, y);
}

async function load(first) {
  try {
    const r = await fetch("data.json?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    document.getElementById("profile").textContent = "profile: " + d.profile;
    const ds = d.dataset || {};
    document.getElementById("dataset").textContent = "dataset: " + (ds.title || ds.name || "?");
    document.getElementById("dataset").title = (ds.description || "") + (ds.groups ? "\nGroups: " + ds.groups.join(", ") : "");
    document.getElementById("reports").replaceChildren(...(d.reports || []).map(r =>
      el("a", { href: r.href, target: "_blank", rel: "noopener", class: "badge", title: "updated " + hhmm(r.updated) },
        `${r.ws.toUpperCase()} PDF`)));
    document.getElementById("built").textContent = "updated " + hhmm(d.built);
    if (first || d.built !== S.built) { S.data = d; S.built = d.built; render(); }
  } catch (e) {
    document.getElementById("built").textContent = "could not load data.json (" + e.message + ")";
    console.error(e);
    const why = e.message === "404" ? "data.json missing: run `alexandria dashboard build`."
      : location.protocol === "file:" ? "Opening the file directly (file://) blocks data.json: use `alexandria dashboard serve`."
      : `dashboard error: ${e.message} (see the browser console).`;
    if (first) document.getElementById("main").replaceChildren(el("p", { class: "err" }, why));
  }
}
load(true);
setInterval(() => { if (!document.hidden) load(false); }, 30000);
