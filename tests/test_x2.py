"""X2 pieces that need no GPU / network: HTML text extraction, gpt-oss output parsing, month windows."""


def test_html_text_keeps_article_drops_bibliography_and_scripts():
    from alexandria.x2.common import html_text
    html = ("<html><nav>menu</nav><article><h1>Title</h1><p>We found <math alttext='x^2'><mi>x</mi></math> grows."
            "</p><script>var a=1;</script><section class='ltx_bibliography'><p>[1] Ref</p></section></article>"
            "<footer>foot</footer></html>")
    t = html_text(html)
    assert "Title" in t and "We found" in t and "x^2" in t and "grows" in t
    assert "menu" not in t and "var a" not in t and "Ref" not in t and "foot" not in t


def test_gpt_oss_final_channel_parsing():
    from alexandria.x2.common import final_text, parse_final_json
    raw = 'analysisWe need {"draft": 1} things.assistantfinal{"questions": [{"question": "Q?", "answer": "A"}]}'
    assert final_text(raw).startswith('{"questions"')
    assert parse_final_json(raw)["questions"][0]["answer"] == "A"
    harmony = '<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final<|message|>{"a": 2}<|return|>'
    assert parse_final_json(harmony) == {"a": 2}
    assert parse_final_json("no json here") is None


def test_month_windows_cover_the_span():
    from alexandria.x2.papers import month_windows
    w = month_windows("2026-06-15", "2026-08-03")
    assert w == [("2026-06-15", "2026-06-30"), ("2026-07-01", "2026-07-31"), ("2026-08-01", "2026-08-03")]
