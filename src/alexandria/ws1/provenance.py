"""Stage `provenance` (reduce): how the summaries were made. Summarizer model(s), prompt, temperature, token totals and
an API-cost estimate, the prompt overhead, and the generation failure rate.

Measured from the data: `summary_model`, `summary_tokens_in/out` (totals, per model), the prompt overhead (the smallest
input token counts are the prompt alone plus a near-empty paper), failure rate (KU-Fixes needs_regeneration plus the
`features` flags refusal / bad claims JSON / output cap). Not in the data: the prompt text and temperature; they are
read from ws1.provenance.prompt_file / temperature and reported as "not published" until filled in.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from ..runtime import Context, stage
from ..stats import wilson


@stage("ws1", "provenance", needs=("features",), config_keys=("ws1.provenance", "ws1.checks"),
       help="summarizer model, prompt, temperature, tokens, cost estimate, prompt overhead, failure rate")
def run(ctx: Context) -> dict:
    pv = ctx.section("ws1.provenance")
    con = duckdb.connect()
    parts = str(ctx.input_dir("features") / "parts" / "*.parquet")
    con.sql(f"create view f as select * from read_parquet('{parts}', union_by_name=true)")
    n = con.sql("select count(*) from f").fetchone()[0]
    models = con.sql("""select coalesce(summary_model,'(none)') m, count(*) n, sum(tokens_in) tin, sum(tokens_out) tout,
                        quantile_cont(tokens_in, 0.5) med_in, quantile_cont(tokens_out, 0.5) med_out
                        from f group by 1 order by 2 desc""").df()
    # prompt overhead: the low tail of tokens_in (prompt + a near-empty paper)
    q = con.sql("select min(tokens_in), quantile_cont(tokens_in, 0.001), quantile_cont(tokens_in, 0.01) "
                "from f where tokens_in > 0").fetchone()
    overhead_est = q[0]
    configured = ctx.section("ws1.checks").get("prompt_overhead_tokens")

    price = pv.get("price_per_mtok") or {}
    scale = 1.0
    if ctx.cfg.profile != "full":            # extrapolate the sample to the local release size
        try:
            import json
            rec = json.loads((ctx.cfg.stage_dir("ws1", "reconcile") / "reconcile.json").read_text())
            scale = sum(v["rows"] for v in rec["local"].values()) / max(n, 1)
        except FileNotFoundError:
            scale = 1.0
    tin, tout = float(models.tin.sum() or 0), float(models.tout.sum() or 0)
    cost = None
    if price.get("input") is not None and price.get("output") is not None:
        cost = (tin * price["input"] + tout * price["output"]) / 1e6 * scale

    fail_sql = {
        "needs regeneration (KU-Fixes)": "ku_needs_regeneration",
        "refusal in a field": "refusal",
        "claims JSON unparsable": "claims_bad_json",
        "output-token cap reached": "output_capped",
        "any of the above": "ku_needs_regeneration or refusal or claims_bad_json or output_capped",
    }
    fails = []
    for label, sql in fail_sql.items():
        k = con.sql(f"select count(*) from f where {sql}").fetchone()[0]
        p, lo, hi = wilson(k, n)
        fails.append({"failure": label, "k": k, "rate": p, "lo": lo, "hi": hi})

    prompt_file = pv.get("prompt_file")
    prompt_text = Path(prompt_file).read_text() if prompt_file and Path(prompt_file).exists() else None
    if prompt_text:
        (ctx.out / "prompt.txt").write_text(prompt_text)
    result = {"papers": n, "models": models.to_dict("records"), "prompt_overhead_estimate": overhead_est,
              "prompt_overhead_quantiles": {"min": q[0], "p0.1": q[1], "p1": q[2]},
              "prompt_overhead_configured": configured, "tokens_in": tin * scale, "tokens_out": tout * scale,
              "scale": scale, "cost_usd_estimate": cost, "price": price, "failures": fails,
              "prompt_published": prompt_text is not None, "temperature": pv.get("temperature")}
    ctx.write_json("provenance.json", result)
    gaps = [g for g, missing in (("generation prompt", prompt_text is None),
                                 ("temperature", pv.get("temperature") is None)) if missing]
    ctx.publish("Generation provenance", [
        {"type": "kpis", "items": [
            {"label": "summarizer models", "value": len(models), "fmt": "int", "hint": ", ".join(models.m)},
            {"label": "input tokens" + (" (extrapolated)" if scale != 1 else ""), "value": tin * scale, "fmt": "int"},
            {"label": "output tokens" + (" (extrapolated)" if scale != 1 else ""), "value": tout * scale, "fmt": "int"},
            {"label": "API cost estimate (USD)", "value": cost, "fmt": "int",
             "hint": f"{price.get('input')}/{price.get('output')} USD per M tokens; {price.get('source', '')}"},
            {"label": "prompt overhead (tokens)", "value": overhead_est, "fmt": "int",
             "hint": f"configured {configured}; p0.1 {q[1]:.0f}" if q[1] else ""},
            {"label": "generation failures", "value": fails[-1]["rate"], "fmt": "pct1",
             "status": "warn" if fails[-1]["rate"] > 0.005 else "good"},
        ]},
        {"type": "table", "title": "Summarizer models", "columns": [
            {"key": "m", "label": "model"}, {"key": "n", "label": "papers", "fmt": "int"},
            {"key": "med_in", "label": "median tokens in", "fmt": "int"},
            {"key": "med_out", "label": "median tokens out", "fmt": "int"}], "rows": models.to_dict("records")},
        {"type": "table", "title": "Failure rate", "hint": "Wilson 95% intervals", "columns": [
            {"key": "failure", "label": "failure"}, {"key": "k", "label": "papers", "fmt": "int"},
            {"key": "rate", "label": "rate", "fmt": "pct1"}, {"key": "lo", "label": "lo", "fmt": "pct1"},
            {"key": "hi", "label": "hi", "fmt": "pct1"}], "rows": fails},
        {"type": "markdown", "text": ("**Still to publish:** " + ", ".join(gaps) +
                                      ". Set `ws1.provenance.prompt_file` and `temperature` in configs/ws1.yaml.")
         if gaps else "Prompt and temperature are recorded (`prompt.txt`)."},
    ], summary="Model, tokens and failures are measured; prompt and temperature come from the config.")
    return {"papers": n, "models": len(models), "cost_usd_estimate": cost}
