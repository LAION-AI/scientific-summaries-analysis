# Dashboard panel contract

Every stage calls `ctx.publish(title, panels, summary="")`, which writes `<stage dir>/dashboard.json`.
`alexandria dashboard build` collects all of them (plus every `manifest*.json`) into one static site; the app renders
one tab per workstream, one section per stage, panels in order. Panels are plain JSON:

| type       | fields                                                                                           |
|------------|--------------------------------------------------------------------------------------------------|
| `kpis`     | `items: [{label, value, fmt?, hint?, status?}]`  status: good / warn / bad colours the tile      |
| `table`    | `title, hint?, columns: [{key, label, fmt?}], rows: [{...}], sort?: key, limit?: n`              |
| `bar`      | `title, hint?, labels: [...], series: [{name, data: [...], lo?: [...], hi?: [...]}], stacked?, horizontal?, fmt?` |
| `line`     | `title, hint?, labels: [...], series: [{name, data: [...]}], fmt?`                               |
| `examples` | `title, hint?, items: [{title, tags?: [...], fields: {name: text}}]`  (flagged records to read) |
| `markdown` | `text` (headings, lists, bold, code, links; rendered with a tiny built-in renderer)               |
| `image`    | `title, src` (path relative to the stage dir; build copies it), `alt`                           |

`fmt`: `int` (thousands separators), `pct` (0-1 -> %), `pct1`, `float`, `float3`, `md` (inline markdown: links, bold, code), `text` (default).
`lo` / `hi` on a bar series draw 95% interval whiskers (same units as `data`).
Keep payloads small: aggregate first; at most a few hundred table rows / 50 examples per panel.
