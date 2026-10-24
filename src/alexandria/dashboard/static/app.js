/* Alexandria dashboard: renders data.json (built by `alexandria dashboard build`). Panel types: see PANELS.md.
   Pages: #/overview (one card per workstream), #/<ws>[/<stage>] (stepper + collapsible stage cards), #/runs. */
"use strict";

const S = { data: null, built: null, route: null, charts: new Map(), slots: new Map() };

/* ------------------------------------------------------------------ small utils */
function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } return null; }
function storeJSON(k, v) { if (v === undefined) { try { return JSON.parse(store(k) || "null"); } catch (e) { return null; } } store(k, JSON.stringify(v)); return null; }
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
function hhmm(iso) { if (!iso) return "–"; const d = new Date(iso); return isNaN(d) ? iso : d.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }); }
function ago(iso) {
  if (!iso) return "–"; const d = new Date(iso); if (isNaN(d)) return iso;
  const s = (Date.now() - d) / 1000; if (s < 60) return "just now"; if (s < 3600) return Math.round(s / 60) + " min ago";
  if (s < 86400) return Math.round(s / 3600) + " h ago"; return hhmm(iso);
}
function dur(s) { if (!isNum(s)) return "–"; if (s < 90) return s.toFixed(0) + " s"; if (s < 5400) return (s / 60).toFixed(1) + " min"; return (s / 3600).toFixed(2) + " h"; }
function norm(st) { const s = String(st || "not run"); return s === "not run" ? "pending" : s; }
const STATUS_LABEL = { done: "done", failed: "failed", running: "running", partial: "partial", skipped: "skipped", pending: "not run" };
function chip(st) { const s = norm(st); return el("span", { class: "st " + s }, el("span", { class: "dot " + s }), STATUS_LABEL[s] || s); }

/* ------------------------------------------------------------------ data helpers */
function wsList() { return Object.keys(S.data.workstreams || {}); }
function wsInfo(ws) { return (S.data.ws_info || {})[ws] || { title: ws.toUpperCase(), question: "" }; }
function stagesOf(ws) {
  /* stage entries + the registry's one-line help (shown while a stage has no summary yet) */
  const help = Object.fromEntries((S.data.registry || []).filter(r => r.ws === ws).map(r => [r.name, r.help]));
  return ((S.data.workstreams || {})[ws] || []).map(st => st.help !== undefined ? st : Object.assign(st, { help: help[st.stage] || "" }));
}
function counts(stages) {
  const c = { done: 0, failed: 0, running: 0, skipped: 0, pending: 0 };
  stages.forEach(st => { const s = norm(st.status); if (s === "partial") c.running++; else c[s] = (c[s] || 0) + 1; });
  return c;
}
function wsStatus(ws) {
  const c = counts(stagesOf(ws)), n = stagesOf(ws).length;
  if (c.failed) return "failed"; if (c.running) return "running"; if (n && c.done + c.skipped === n) return "done";
  return c.done ? "partial" : "pending";
}
function lastUpdate(stages) { return stages.map(s => s.updated || s.finished).filter(Boolean).sort().pop(); }
function headline(ws) {
  /* panels flagged {"headline": true} win; else the KPI panel of the last finished stage that has one */
  const stages = stagesOf(ws);
  for (const st of [...stages].reverse()) for (const p of st.panels || []) if (p.headline && p.type === "kpis") return { st, items: p.items || [] };
  for (const st of [...stages].reverse()) {
    if (norm(st.status) !== "done") continue;
    const p = (st.panels || []).find(q => q.type === "kpis" && (q.items || []).length);
    if (p) return { st, items: p.items };
  }
  return null;
}
function layers(ws) {
  const nodes = (S.data.registry || []).filter(r => r.ws === ws), by = Object.fromEntries(nodes.map(n => [n.name, n])), depth = {};
  const d = (n, seen = new Set()) => { if (depth[n.name] != null) return depth[n.name]; if (seen.has(n.name)) return 0; seen.add(n.name);
    const deps = (n.needs || []).map(x => by[x]).filter(Boolean); return (depth[n.name] = deps.length ? 1 + Math.max(...deps.map(x => d(x, seen))) : 0); };
  nodes.forEach(n => d(n));
  /* stages that neither need nor feed another stage (e.g. the PDF report) are shown last, not as inputs */
  const fed = new Set(nodes.flatMap(n => n.needs || [])), isolated = n => !(n.needs || []).length && !fed.has(n.name);
  const max = Math.max(0, ...nodes.filter(n => !isolated(n)).map(n => depth[n.name]));
  nodes.forEach(n => { if (isolated(n) && nodes.length > 1) depth[n.name] = max + 1; });
  const out = []; nodes.forEach(n => (out[depth[n.name]] ||= []).push(n));
  return { nodes, layers: out.filter(Boolean) };
}

/* ------------------------------------------------------------------ series colours: fixed slots per page */
const N_SLOTS = 8;
const PINNED = { "arxiv/new": 1, "arxiv/old": 2, "arxiv/titled": 3, "pubmed": 4, "bethgelab": 5, "web-papers": 7 };
function prepareSlots(panels) {
  const names = [];
  panels.forEach(p => { if (p.type === "bar" || p.type === "line") (p.series || []).forEach(s => { if (!names.includes(s.name)) names.push(s.name); }); });
  S.slots = new Map();
  names.forEach(n => { if (PINNED[n]) S.slots.set(n, PINNED[n]); });
  const used = new Set(S.slots.values());
  names.forEach(n => { if (S.slots.has(n)) return; let k = 1; while (k <= N_SLOTS && used.has(k)) k++; if (k <= N_SLOTS) { S.slots.set(n, k); used.add(k); } });
}
function chartSlots(series) {
  /* colours for one chart: the page-wide slot of each name where it is free in this chart, else the next free slot,
     so the same name keeps its colour across a page and no two series of one chart share one */
  const local = new Map(), used = new Set();
  series.forEach(x => { const k = S.slots.get(x.name); if (k && !used.has(k)) { local.set(x.name, k); used.add(k); } });
  series.forEach(x => { if (local.has(x.name)) return; let k = 1; while (k <= N_SLOTS && used.has(k)) k++; if (k <= N_SLOTS) { local.set(x.name, k); used.add(k); } });
  return name => { const k = local.get(name); return cssVar(k ? `--series-${k}` : "--series-other"); };
}
function foldSeries(series) {
  /* more than N_SLOTS series in one chart: keep the first N_SLOTS - 1, sum the rest into "Other" */
  if (series.length <= N_SLOTS) return series;
  const keep = series.slice(0, N_SLOTS - 1), rest = series.slice(N_SLOTS - 1);
  const n = Math.max(...rest.map(s => (s.data || []).length));
  return [...keep, { name: "Other", data: Array.from({ length: n }, (_, i) => rest.reduce((a, s) => a + (isNum(s.data[i]) ? s.data[i] : 0), 0)) }];
}

/* ------------------------------------------------------------------ theme */
function applyTheme() {
  const t = store("alex-theme") || "auto";
  if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  const b = document.getElementById("theme"); b.textContent = t === "dark" ? "☾" : t === "light" ? "☀" : "◐"; b.title = "Theme: " + t + " (click to change)";
}
document.getElementById("theme").addEventListener("click", () => {
  const order = ["auto", "light", "dark"], t = store("alex-theme") || "auto";
  store("alex-theme", order[(order.indexOf(t) + 1) % 3]); applyTheme(); render(true);
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
function trackChart(key, chart) { if (!S.charts.has(key)) S.charts.set(key, []); S.charts.get(key).push(chart); }
function destroyCharts(key) { (S.charts.get(key) || []).forEach(c => c.destroy()); S.charts.delete(key); }
function destroyAll() { [...S.charts.keys()].forEach(destroyCharts); }

function chartTable(p, series) {
  const cols = [{ key: "_label", label: p.horizontal ? "category" : "x" }];
  series.forEach((s, i) => { cols.push({ key: "s" + i, label: s.name || "value", fmt: p.fmt }); if (s.lo && s.hi) cols.push({ key: "ci" + i, label: (s.name ? s.name + " " : "") + "95% CI" }); });
  const rows = (p.labels || []).map((l, j) => {
    const r = { _label: l };
    series.forEach((s, i) => { r["s" + i] = s.data[j]; if (s.lo && s.hi) r["ci" + i] = `${fmt(s.lo[j], p.fmt)} – ${fmt(s.hi[j], p.fmt)}`; });
    return r;
  });
  return tablePanel({ columns: cols, rows, bare: true });
}

function chartPanel(p, kind, key) {
  const toggle = el("button", { type: "button", class: "more", "aria-pressed": "false" }, "table");
  const card = el("div", { class: "card" + (p.wide ? " wide" : "") },
    el("div", { class: "cardhead" }, el("div", {}, p.title ? el("h3", {}, p.title) : null, p.hint ? el("p", { class: "hint" }, p.hint) : null), toggle));
  const labels = p.labels || [], horiz = !!p.horizontal, many = labels.length > 14;
  const series = foldSeries(p.series || []);
  const box = el("div", { class: "chart" + ((horiz && labels.length > 8) || many ? " tall" : "") });
  const tbl = chartTable(p, series); tbl.hidden = true;
  toggle.addEventListener("click", () => { const show = tbl.hidden; tbl.hidden = !show; box.hidden = show; toggle.setAttribute("aria-pressed", String(show)); toggle.textContent = show ? "chart" : "table"; });
  card.append(box, tbl);
  if (!window.Chart) { box.append(el("p", { class: "empty" }, "Chart.js did not load (offline and vendor file missing): use the table view.")); return card; }
  const canvas = el("canvas", { role: "img", "aria-label": `${p.title || kind} chart; press the table button for the values` });
  box.append(canvas);
  const text = cssVar("--text"), muted = cssVar("--muted"), grid = cssVar("--grid"), surface = cssVar("--panel"), f = p.fmt;
  const colorOf = chartSlots(series);
  const datasets = series.map(s => {
    const col = colorOf(s.name);
    const d = kind === "line"
      ? { label: s.name, data: s.data, borderColor: col, backgroundColor: col, borderWidth: 2, pointRadius: 3, pointHoverRadius: 5, tension: 0.2 }
      : { label: s.name, data: s.data, backgroundColor: col, borderColor: surface, borderWidth: 1, borderRadius: 4, borderSkipped: "start", maxBarThickness: 20, categoryPercentage: 0.75, barPercentage: 0.9 };
    if (s.lo && s.hi) { d._lo = s.lo; d._hi = s.hi; }
    return d;
  });
  const valueAxis = { stacked: !!p.stacked, beginAtZero: kind === "bar", grid: { color: grid }, border: { display: false }, ticks: { color: muted, font: { size: 11 }, callback: v => fmt(v, f) } };
  const catAxis = { stacked: !!p.stacked, grid: { display: false }, border: { color: grid }, ticks: { color: muted, font: { size: 11 }, autoSkip: !horiz, maxRotation: 50, callback(v) { const s = String(this.getLabelForValue(v)); return s.length > 26 ? s.slice(0, 25) + "…" : s; } } };
  /* create the chart once the card is in the document and has a size */
  requestAnimationFrame(() => {
    if (!canvas.isConnected) return;
    const chart = new Chart(canvas, {
      type: kind, data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, indexAxis: horiz ? "y" : "x",
        interaction: kind === "line" ? { mode: "index", intersect: false } : { mode: "nearest", intersect: true },
        scales: horiz ? { x: valueAxis, y: catAxis } : { x: catAxis, y: valueAxis },
        plugins: {
          legend: { display: datasets.length >= 2, position: "bottom", labels: { color: text, boxWidth: 10, boxHeight: 10, usePointStyle: true, pointStyle: kind === "line" ? "circle" : "rectRounded", font: { size: 11 } } },
          tooltip: { titleColor: text, bodyColor: text, backgroundColor: surface, borderColor: grid, borderWidth: 1, padding: 10, cornerRadius: 8,
            callbacks: { label(c) { const ds = c.dataset; let s = (ds.label ? ds.label + ": " : "") + fmt(c.parsed[horiz ? "x" : "y"], f); if (ds._lo) s += `  [${fmt(ds._lo[c.dataIndex], f)}, ${fmt(ds._hi[c.dataIndex], f)}]`; return s; } } },
        },
      },
      plugins: [whiskers],
    });
    trackChart(key, chart);
  });
  return card;
}

/* ------------------------------------------------------------------ panel renderers */
function kpisPanel(p) {
  return el("div", { class: "kpis wide", style: "grid-column:1/-1" }, (p.items || []).map(it =>
    el("div", { class: "kpi " + (it.status || ""), title: it.hint || "" }, el("div", { class: "v" }, fmt(it.value, it.fmt)), el("div", { class: "l" }, it.label),
      it.status ? el("div", { class: "h" }, it.status === "good" ? "OK" : it.status === "warn" ? "check" : "problem") : null,
      it.hint ? el("div", { class: "h" }, it.hint) : null)));
}

function tablePanel(p) {
  const cols = p.columns || (p.rows && p.rows.length ? Object.keys(p.rows[0]).map(k => ({ key: k, label: k })) : []);
  const all = (p.rows || []).slice();
  let sortKey = p.sort || null, dir = p.sort ? -1 : 1, page = 0, showAll = false, q = "";
  const size = p.pageSize || 10;
  const wrap = el("div", { class: "tablewrap" }), pager = el("div", { class: "pager" });
  const box = el("div", {});
  const isNumCol = c => all.some(r => isNum(r[c.key])) && !all.some(r => r[c.key] != null && !isNum(r[c.key]));
  const cell = (r, c) => {
    if (c.fmt !== "md") return el("td", { class: isNum(r[c.key]) ? "num" : null }, fmt(r[c.key], c.fmt));
    const td = el("td", {}); td.innerHTML = mdInline(esc(r[c.key] ?? "")); return td;
  };
  const search = all.length > 10 ? el("input", { type: "search", placeholder: `Search ${all.length} rows…`, "aria-label": "search table",
    oninput: e => { q = e.target.value.toLowerCase(); page = 0; draw(); } }) : null;
  const count = el("span", { class: "n" });
  function draw() {
    let rows = q ? all.filter(r => cols.some(c => String(r[c.key] ?? "").toLowerCase().includes(q))) : all.slice();
    if (sortKey) rows.sort((a, b) => { const x = a[sortKey], y = b[sortKey]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    const pages = Math.max(1, Math.ceil(rows.length / size)); page = Math.min(page, pages - 1);
    const shown = showAll ? rows : rows.slice(page * size, page * size + size);
    const head = el("tr", {}, cols.map(c => el("th", { scope: "col", tabindex: "0", "aria-sort": sortKey === c.key ? (dir > 0 ? "ascending" : "descending") : null,
      onclick: () => { if (sortKey === c.key) dir = -dir; else { sortKey = c.key; dir = isNumCol(c) ? -1 : 1; } draw(); },
      onkeydown: e => { if (e.key === "Enter") e.target.click(); } }, c.label || c.key)));
    wrap.replaceChildren(el("table", {}, el("thead", {}, head), el("tbody", {}, shown.map(r => el("tr", {}, cols.map(c => cell(r, c)))))));
    if (!rows.length) wrap.append(el("p", { class: "empty pad" }, q ? "no matching rows" : "no rows"));
    count.textContent = q ? `${rows.length} of ${all.length} rows` : `${all.length} rows`;
    pager.replaceChildren();
    if (rows.length > size) {
      if (!showAll) pager.append(
        el("button", { type: "button", disabled: page === 0 || null, onclick: () => { page--; draw(); } }, "‹ Prev"),
        el("span", {}, `${page + 1} / ${pages}`),
        el("button", { type: "button", disabled: page >= pages - 1 || null, onclick: () => { page++; draw(); } }, "Next ›"));
      pager.append(el("button", { type: "button", onclick: () => { showAll = !showAll; draw(); } }, showAll ? "Paginate" : "Show all"));
    }
  }
  draw();
  if (search || all.length > size) box.append(el("div", { class: "tabletools" }, search || el("span", {}), count));
  box.append(wrap, pager);
  if (p.bare) return box;
  const wide = cols.length > 5 || p.wide;
  return el("div", { class: "card" + (wide ? " wide" : "") }, p.title ? el("h3", {}, p.title) : null, p.hint ? el("p", { class: "hint" }, p.hint) : null, box);
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
  if (p.missing) { card.append(el("p", { class: "empty" }, `image missing: ${p.src}`)); return card; }
  card.append(el("img", { class: "panel", src: p.src, alt: p.alt || p.title || "figure", loading: "lazy", onclick: () => zoom(p.src, p.alt || p.title) }));
  return card;
}
function zoom(src, alt) {
  const lb = document.getElementById("lightbox"), img = lb.querySelector("img");
  img.src = src; img.alt = alt || ""; lb.hidden = false;
}
document.getElementById("lightbox").addEventListener("click", e => { e.currentTarget.hidden = true; });
document.addEventListener("keydown", e => { if (e.key === "Escape") document.getElementById("lightbox").hidden = true; });

function panel(p, key) {
  switch (p.type) {
    case "kpis": return kpisPanel(p);
    case "table": return tablePanel(p);
    case "bar": return chartPanel(p, "bar", key);
    case "line": return chartPanel(p, "line", key);
    case "examples": return examplesPanel(p);
    case "markdown": return markdownPanel(p);
    case "image": return imagePanel(p);
    default: return el("div", { class: "card" }, el("p", { class: "empty" }, `unknown panel type: ${p.type}`));
  }
}

/* ------------------------------------------------------------------ navigation */
function go(route) { if (location.hash !== "#/" + route) location.hash = "#/" + route; else render(); }
function parseRoute() {
  const h = decodeURIComponent(location.hash.replace(/^#\/?/, "")) || store("alex-route") || "overview";
  const [page, stage] = h.split("/");
  const ok = page === "overview" || page === "runs" || wsList().includes(page);
  return ok ? { page, stage: stage || null } : { page: "overview", stage: null };
}
function sidebar(route) {
  const item = (page, code, label, extra, dotStatus) => el("button", { type: "button", class: "navitem" + (route.page === page ? " on" : ""), "aria-current": route.page === page ? "page" : null,
    onclick: () => go(page), title: label }, dotStatus ? el("span", { class: "dot " + norm(dotStatus) }) : null, el("span", { class: "code" }, code), el("span", { class: "lbl" }, label), extra);
  const nav = document.getElementById("nav");
  nav.replaceChildren(
    item("overview", "◎", "Overview"),
    el("div", { class: "sec" }, "Workstreams"),
    ...wsList().map(ws => { const st = stagesOf(ws), c = counts(st); return item(ws, ws.toUpperCase(), wsInfo(ws).title, el("span", { class: "count" }, `${c.done}/${st.length}`), wsStatus(ws)); }),
    el("div", { class: "sec" }, "Monitoring"),
    item("runs", "≡", "Runs", el("span", { class: "count" }, String((S.data.runs || []).length))));
}

/* ------------------------------------------------------------------ overview */
function progressBar(c, n) {
  const seg = k => c[k] ? el("span", { class: k, style: `width:${(100 * c[k] / Math.max(n, 1)).toFixed(2)}%`, title: `${c[k]} ${k}` }) : null;
  return el("div", { class: "progress", role: "img", "aria-label": `${c.done} of ${n} stages done` }, seg("done"), seg("running"), seg("failed"), seg("skipped"));
}
function legend(c) {
  return el("div", { class: "legend" }, ["done", "running", "failed", "skipped", "pending"].filter(k => c[k]).map(k => el("span", {}, el("span", { class: "dot " + k }), `${c[k]} ${STATUS_LABEL[k]}`)));
}
function flow(ws) {
  const { layers: ls } = layers(ws), out = el("div", { class: "flow" });
  const status = Object.fromEntries(stagesOf(ws).map(s => [s.stage, s.status]));
  ls.forEach((layer, i) => {
    if (i) out.append(el("span", { class: "arrow", "aria-hidden": "true" }, "→"));
    out.append(el("div", { class: "grp" }, layer.map(n => el("a", { class: "pill", href: `#/${ws}/${encodeURIComponent(n.name)}`, title: n.help || n.name },
      el("span", { class: "dot " + norm(status[n.name] || n.status) }), n.name))));
  });
  return out;
}
function wsCard(ws) {
  const info = wsInfo(ws), st = stagesOf(ws), c = counts(st), h = headline(ws);
  const rep = (S.data.reports || []).find(r => r.ws === ws);
  return el("article", { class: "wscard" },
    el("div", { class: "top" }, el("div", { class: "wsbadge" }, ws.toUpperCase()),
      el("div", { style: "flex:1;min-width:0" }, el("h2", {}, el("a", { href: `#/${ws}`, style: "color:inherit" }, info.title)), info.question ? el("p", { class: "q" }, info.question) : null),
      chip(wsStatus(ws))),
    progressBar(c, st.length), legend(c),
    h ? el("div", { class: "minikpis", title: `from stage “${h.st.title || h.st.stage}”` }, h.items.slice(0, 4).map(it =>
      el("div", { class: "minikpi" }, el("div", { class: "v" }, fmt(it.value, it.fmt)), el("div", { class: "l" }, it.label)))) :
      el("p", { class: "empty", style: "margin:0" }, "no results yet"),
    flow(ws),
    el("div", { class: "foot" }, el("span", {}, lastUpdate(st) ? "updated " + ago(lastUpdate(st)) : "not started yet"), el("span", { class: "spacer" }),
      rep ? el("a", { class: "btn small", href: rep.href, target: "_blank", rel: "noopener" }, "PDF report") : null,
      el("a", { class: "btn small", href: `#/${ws}` }, "Open →")));
}
function overviewPage() {
  const reg = S.data.registry || [], all = wsList().flatMap(stagesOf), c = counts(all);
  const stat = (v, l, cls) => el("div", { class: "stat " + (cls || "") }, el("div", { class: "v" }, fmt(v, "int")), el("div", { class: "l" }, l));
  const ds = S.data.dataset || {};
  return [
    el("div", { class: "pagehead" }, el("div", { class: "t" }, el("div", { class: "eyebrow" }, "Overview"),
      el("h1", {}, "Alexandria: are structured paper summaries a better knowledge source?"),
      el("p", {}, `${wsList().length} workstreams on ${ds.title || ds.name || "the dataset"} (profile ${S.data.profile}). Each card is one research question; click a stage to see its results.`))),
    el("div", { class: "stats" }, stat(reg.length || all.length, "stages"), stat(c.done, "done", "good"), stat(c.running, "running", c.running ? "run" : ""),
      stat(c.failed, "failed", c.failed ? "bad" : ""), stat(c.pending, "not run yet"), stat((S.data.runs || []).length, "job runs")),
    el("div", { class: "wsgrid" }, wsList().map(wsCard)),
  ];
}

/* ------------------------------------------------------------------ workstream page */
function openSet(ws) {
  const saved = storeJSON("alex-open-" + ws);
  if (Array.isArray(saved)) return new Set(saved);
  const st = stagesOf(ws), out = new Set(st.filter(s => norm(s.status) === "failed").map(s => s.stage));
  /* the latest finished stage with real results (KPIs, tables, charts, figures), not one that only has notes */
  const rich = s => (s.panels || []).some(p => p.type !== "markdown");
  const latest = st.filter(s => norm(s.status) === "done" && rich(s)).sort((a, b) => String(a.updated || "").localeCompare(String(b.updated || ""))).pop();
  if (latest) out.add(latest.stage);
  return out;
}
function saveOpen(ws, set) { storeJSON("alex-open-" + ws, [...set]); }

function stageBody(st) {
  const key = `${st.ws}/${st.stage}`, reg = (S.data.registry || []).find(r => r.ws === st.ws && r.name === st.stage) || {};
  const body = el("div", { class: "body" });
  const help = [reg.help, reg.kind && reg.kind !== "?" && `${reg.kind} · ${reg.resources} · env ${reg.env}`, (reg.needs || []).length ? "needs: " + reg.needs.join(", ") : null].filter(Boolean).join("  ·  ");
  if (help) body.append(el("p", { class: "help" }, help));
  if (st.summary && String(st.summary).length > 120) body.append(el("div", { class: "md", html: markdown(st.summary) }));
  const grid = el("div", { class: "grid" });
  (st.panels || []).forEach(p => grid.append(panel(p, key)));
  if (!(st.panels || []).length) grid.append(norm(st.status) === "failed"
    ? el("p", {}, "This stage failed before publishing results. ", el("a", { href: "#/runs" }, "Traceback in Runs →"))
    : el("p", { class: "empty" }, norm(st.status) === "pending" ? "Not run yet." : "This stage writes data for later stages; it publishes no panels."));
  body.append(grid);
  return body;
}
function stageCard(st, open, onToggle) {
  const s = norm(st.status), key = `${st.ws}/${st.stage}`;
  const card = el("section", { class: "stage " + s + (open ? " open" : ""), id: `s-${st.ws}-${st.stage}` });
  const head = el("button", { type: "button", class: "sh", "aria-expanded": String(open) },
    el("span", { class: "chev", "aria-hidden": "true" }, "›"),
    el("div", { style: "min-width:0" },
      el("div", { class: "ttl" }, el("b", {}, st.title || st.stage), el("span", { class: "id" }, key)),
      (st.summary || st.help) ? el("div", { class: "sum", title: st.summary || st.help }, String(st.summary || st.help).replace(/[*`#]/g, "")) : null,
      st.error ? el("div", { class: "errline" }, st.error) : null),
    el("div", { class: "meta" }, chip(st.status), st.seconds != null ? el("span", { title: "compute time (all tasks)" }, dur(st.seconds)) : null,
      el("span", { title: st.updated || st.finished || st.started || "" }, ago(st.updated || st.finished || st.started))));
  card.append(head);
  let body = open ? stageBody(st) : null; if (body) card.append(body);
  head.addEventListener("click", () => {
    const now = !card.classList.contains("open");
    card.classList.toggle("open", now); head.setAttribute("aria-expanded", String(now));
    if (now) { body = stageBody(st); card.append(body); } else { destroyCharts(key); body && body.remove(); body = null; }
    onToggle(st.stage, now);
  });
  return card;
}
function pipelineGraph(ws) {
  const { nodes, layers: ls } = layers(ws), status = Object.fromEntries(stagesOf(ws).map(s => [s.stage, s.status]));
  const wrap = el("div", { class: "dag" }), svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "edges"); svg.setAttribute("aria-hidden", "true"); wrap.append(svg);
  ls.forEach((layer, i) => wrap.append(el("div", { class: "layer" }, el("div", { class: "lt" }, i === 0 ? "inputs" : `step ${i}`),
    layer.map(n => el("a", { class: "node", href: `#/${ws}/${encodeURIComponent(n.name)}`, "data-name": n.name, title: n.help || "" },
      el("span", { class: "dot " + norm(status[n.name] || n.status) }), n.name)))));
  const draw = () => {
    if (!wrap.isConnected) return;
    const box = wrap.getBoundingClientRect(); svg.replaceChildren();
    svg.setAttribute("width", wrap.scrollWidth); svg.setAttribute("height", wrap.scrollHeight);
    nodes.forEach(n => (n.needs || []).forEach(dep => {
      const a = wrap.querySelector(`[data-name="${CSS.escape(dep)}"]`), b = wrap.querySelector(`[data-name="${CSS.escape(n.name)}"]`); if (!a || !b) return;
      const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
      const x1 = ra.right - box.left + wrap.scrollLeft, y1 = ra.top + ra.height / 2 - box.top, x2 = rb.left - box.left + wrap.scrollLeft, y2 = rb.top + rb.height / 2 - box.top;
      const p = document.createElementNS("http://www.w3.org/2000/svg", "path"), mx = (x1 + x2) / 2;
      p.setAttribute("d", `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2 - 2},${y2}`); svg.append(p);
    }));
  };
  const det = el("details", { class: "pipe", ontoggle: () => requestAnimationFrame(draw) }, el("summary", {}, "Pipeline graph"), wrap);
  if (store("alex-pipe") === "1") det.open = true;
  det.addEventListener("toggle", () => store("alex-pipe", det.open ? "1" : "0"));
  window.addEventListener("resize", () => requestAnimationFrame(draw), { passive: true });
  return det;
}
function wsPage(ws, focus) {
  const info = wsInfo(ws), st = stagesOf(ws), c = counts(st), open = openSet(ws);
  if (focus) open.add(focus);
  const rep = (S.data.reports || []).find(r => r.ws === ws);
  prepareSlots(st.flatMap(s => s.panels || []));
  const list = el("div", { class: "stages" });
  const cards = new Map();
  const onToggle = (name, isOpen) => { if (isOpen) open.add(name); else open.delete(name); saveOpen(ws, open); stepper.querySelectorAll(".step").forEach(b => b.classList.toggle("sel", open.has(b.dataset.name))); };
  st.forEach(s => { const card = stageCard(s, open.has(s.stage), onToggle); cards.set(s.stage, card); list.append(card); });
  const stepper = el("div", { class: "stepper", role: "list" }, st.map((s, i) => el("button", { type: "button", role: "listitem", class: "step " + norm(s.status) + (open.has(s.stage) ? " sel" : ""), "data-name": s.stage, title: `${s.stage}: ${STATUS_LABEL[norm(s.status)]}`,
    onclick: () => { const card = cards.get(s.stage); if (!card.classList.contains("open")) card.querySelector(".sh").click(); card.scrollIntoView({ behavior: "smooth", block: "start" }); } },
    el("span", { class: "n" }, String(i + 1)), s.stage)));
  const setAll = on => { cards.forEach(card => { if (card.classList.contains("open") !== on) card.querySelector(".sh").click(); }); };
  if (focus) requestAnimationFrame(() => { const card = cards.get(focus); if (card) card.scrollIntoView({ block: "start" }); });
  return [
    el("div", { class: "pagehead" },
      el("div", { class: "t" }, el("div", { class: "eyebrow" }, ws.toUpperCase()), el("h1", {}, info.title), info.question ? el("p", {}, info.question) : null,
        el("div", { style: "margin-top:10px" }, legend(c))),
      el("div", { class: "actions" }, rep ? el("a", { class: "btn", href: rep.href, target: "_blank", rel: "noopener" }, "PDF report") : null,
        el("button", { type: "button", class: "btn", onclick: () => setAll(true) }, "Expand all"),
        el("button", { type: "button", class: "btn", onclick: () => setAll(false) }, "Collapse all"))),
    stepper, pipelineGraph(ws), list,
  ];
}

/* ------------------------------------------------------------------ runs page */
function runsPage() {
  const runs = (S.data.runs || []).map(r => ({ ...r, task: r.shard ? `${r.shard[0] + 1}/${r.shard[1]}` : "" }));
  const q = el("input", { type: "search", placeholder: "Filter by stage, host, job, error…", "aria-label": "filter runs", value: store("alex-runq") || "" });
  const sel = el("select", { "aria-label": "status" }, ["all", "done", "failed", "running", "skipped"].map(s => el("option", { value: s }, s)));
  sel.value = store("alex-runst") || "all";
  const body = el("div", {});
  let sortKey = "started", dir = -1, page = 0;
  const size = 25;
  const cols = [["started", "started"], ["ws", "ws"], ["stage", "stage"], ["status", "status"], ["task", "task"], ["seconds", "time"], ["host", "host"], ["slurm_job", "job"], ["error", "error"]];
  function draw() {
    store("alex-runq", q.value); store("alex-runst", sel.value);
    const t = q.value.toLowerCase();
    let rows = runs.filter(r => (sel.value === "all" || r.status === sel.value) && (!t || JSON.stringify([r.ws, r.stage, r.host, r.error, r.slurm_job]).toLowerCase().includes(t)));
    rows.sort((a, b) => { const x = a[sortKey], y = b[sortKey]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    const pages = Math.max(1, Math.ceil(rows.length / size)); page = Math.min(page, pages - 1);
    const shown = rows.slice(page * size, page * size + size);
    const head = el("tr", {}, cols.map(([k, l]) => el("th", { scope: "col", "aria-sort": sortKey === k ? (dir > 0 ? "ascending" : "descending") : null,
      onclick: () => { if (sortKey === k) dir = -dir; else { sortKey = k; dir = k === "started" || k === "seconds" ? -1 : 1; } draw(); } }, l)));
    const tb = el("tbody", {});
    shown.forEach(r => {
      const tr = el("tr", {}, el("td", {}, hhmm(r.started)), el("td", {}, r.ws), el("td", {}, el("a", { href: `#/${r.ws}/${encodeURIComponent(r.stage)}` }, r.stage)), el("td", {}, chip(r.status)),
        el("td", {}, r.task), el("td", { class: "num" }, dur(r.seconds)), el("td", {}, r.host || "–"), el("td", {}, r.slurm_job || "–"),
        el("td", { class: "err" }, r.error || "", r.traceback ? el("span", {}, " ", el("button", { type: "button", class: "linkbtn", onclick: e => {
          const nx = tr.nextElementSibling; if (nx && nx.classList.contains("trace")) { nx.remove(); e.target.textContent = "traceback"; return; }
          tr.after(el("tr", { class: "trace" }, el("td", { colspan: cols.length }, el("pre", {}, r.traceback)))); e.target.textContent = "hide"; } }, "traceback")) : null));
      tb.append(tr);
    });
    const pager = el("div", { class: "pager" }, el("span", {}, `${rows.length} runs`),
      el("button", { type: "button", disabled: page === 0 || null, onclick: () => { page--; draw(); } }, "‹ Prev"), el("span", {}, `${page + 1} / ${pages}`),
      el("button", { type: "button", disabled: page >= pages - 1 || null, onclick: () => { page++; draw(); } }, "Next ›"));
    body.replaceChildren(el("div", { class: "tablewrap", style: "max-height:none" }, el("table", {}, el("thead", {}, head), tb)), pager);
  }
  q.addEventListener("input", () => { page = 0; draw(); }); sel.addEventListener("change", () => { page = 0; draw(); });
  draw();
  return [el("div", { class: "pagehead" }, el("div", { class: "t" }, el("div", { class: "eyebrow" }, "Monitoring"), el("h1", {}, "Runs"),
    el("p", {}, "Every stage execution (one row per SLURM task) with its host, job id and error. Results folder: ", el("code", {}, S.data.out_root || "")))),
    el("div", { class: "toolbar" }, q, sel), body];
}

/* ------------------------------------------------------------------ shell */
function render(keepScroll) {
  if (!S.data) return;
  const route = parseRoute(), y = window.scrollY;
  const same = S.route && S.route.page === route.page && S.route.stage === route.stage;
  S.route = route; store("alex-route", route.page);
  destroyAll();
  sidebar(route);
  const main = document.getElementById("main");
  const content = route.page === "overview" ? overviewPage() : route.page === "runs" ? runsPage() : wsPage(route.page, route.stage);
  main.replaceChildren(...content);
  if (keepScroll || same) window.scrollTo(0, y); else if (!route.stage) window.scrollTo(0, 0);
}
window.addEventListener("hashchange", () => render(false));

async function load(first) {
  try {
    const r = await fetch("data.json?t=" + Date.now(), { cache: "no-store" });
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    document.getElementById("profile").textContent = "profile: " + d.profile;
    const ds = d.dataset || {};
    document.getElementById("dataset").textContent = ds.title || ds.name || "dataset";
    document.getElementById("dataset").title = (ds.description || "") + (ds.groups ? "\nGroups: " + ds.groups.join(", ") : "");
    document.getElementById("reports").replaceChildren(...(d.reports || []).map(r =>
      el("a", { href: r.href, target: "_blank", rel: "noopener", title: "PDF report, updated " + hhmm(r.updated) }, `${r.ws.toUpperCase()} PDF`)));
    document.getElementById("built").replaceChildren(el("span", { class: "live", "aria-hidden": "true" }), "updated " + hhmm(d.built));
    if (first || d.built !== S.built) { S.data = d; S.built = d.built; render(!first); }
  } catch (e) {
    document.getElementById("built").textContent = "could not load data.json (" + e.message + ")";
    console.error(e);
    const why = e.message === "404" ? "data.json missing: run `alexandria dashboard build`."
      : location.protocol === "file:" ? "Opening the file directly (file://) blocks data.json: use `alexandria dashboard serve`."
      : `dashboard error: ${e.message} (see the browser console).`;
    if (first) document.getElementById("main").replaceChildren(el("p", { class: "err pad" }, why));
  }
}
load(true);
setInterval(() => { if (!document.hidden) load(false); }, 30000);
