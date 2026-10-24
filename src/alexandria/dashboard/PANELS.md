# Dashboard panel contract

Every stage calls `ctx.publish(title, panels, summary="")`, which writes `<stage dir>/dashboard.json`.
`alexandria dashboard build` collects all of them (plus every `manifest*.json`) into one static site (`data.json`).

Pages: **Overview** (one card per workstream: research question, stage progress, 2-4 headline KPIs, compact pipeline;
click a stage pill to jump to it), **one page per workstream** (stepper + one collapsible card per stage; collapsed
cards show status, one-line `summary`, compute time, last update and, for failed stages, the error), **Runs** (every
manifest, searchable, tracebacks on demand). Routes: `#/overview`, `#/<ws>`, `#/<ws>/<stage>`, `#/runs`.

Write for the collapsed card: `summary` is the one line everybody reads (e.g. "418 questions x 5 arms x 4 models");
put the stage's key numbers in its first `kpis` panel. The workstream's one-line question comes from `WS_INFO` in
`dashboard/site.py` (add an entry for a new WS; otherwise it is titled by its id).

**Headline KPIs** (Overview card): a `kpis` panel with `"headline": true` (any stage of the WS; the last one in
pipeline order wins); otherwise the first `kpis` panel of the last finished stage. Mark the 2-4 numbers that answer
the WS's question, e.g. `{"type": "kpis", "headline": true, "items": [...]}`.

Panels are plain JSON:

| type       | fields                                                                                           |
|------------|--------------------------------------------------------------------------------------------------|
| `kpis`     | `items: [{label, value, fmt?, hint?, status?}]`, `headline?: true`  status: good / warn / bad colours the tile |
| `table`    | `title, hint?, columns: [{key, label, fmt?}], rows: [{...}], sort?: key, pageSize?: n` (search box above 10 rows, 10 rows per page, "Show all") |
| `bar`      | `title, hint?, labels: [...], series: [{name, data: [...], lo?: [...], hi?: [...]}], stacked?, horizontal?, fmt?` |
| `line`     | `title, hint?, labels: [...], series: [{name, data: [...]}], fmt?`                               |
| `examples` | `title, hint?, items: [{title, tags?: [...], fields: {name: text}}]`  (flagged records to read) |
| `markdown` | `text` (headings, lists, bold, code, links; rendered with a tiny built-in renderer)               |
| `image`    | `title, src` (path relative to the stage dir; build copies it), `alt`; click to zoom             |

`fmt`: `int` (thousands separators), `pct` (0-1 -> %), `pct1`, `float`, `float3`, `md` (inline markdown: links, bold, code), `text` (default).
`lo` / `hi` on a bar series draw 95% interval whiskers (same units as `data`).
Keep payloads small: aggregate first; at most a few hundred table rows / 50 examples per panel.

Charts: 240 px high (340 px with many categories); each series name keeps one colour on a workstream page.
