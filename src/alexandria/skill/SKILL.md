---
name: alexandria-skill
description: Look up findings, numbers, methods and limitations from scientific papers in the Alexandria corpus (one knowledge unit per paper: its structured summary; plus open-access full text). Use when a question needs specific evidence from the scientific literature.
---

# Alexandria knowledge skill

Tools (MCP server `alexandria skill mcp`, or OpenAI function schemas from `alexandria.skill.tools.openai_tools`):

| tool | use it to |
|---|---|
| `search(query, field="any", k=5)` | find papers whose structured summary matches; `field` narrows to `claims`, `key_results`, `methods`, `executive_summary` or `limitations` |
| `get_paper(id, fields)` | read chosen summary fields of one paper (ids come from search results) |
| `search_fulltext(query, k=5)` | find passages in the full text of open-access papers |

How to use it well:
1. Start with `search` using the distinctive terms of the question (system, method, quantity), not the whole question.
2. Narrow with `field="key_results"` for numbers and findings, `field="methods"` for how something was done,
   `field="limitations"` for caveats.
3. Open the 1-2 most promising papers with `get_paper` before answering; check that the paper is about the same
   system and setting as the question.
4. Answer with the specific value or phrase the evidence supports. If nothing relevant is found after a few
   searches, say so instead of guessing.

Every response is cached by (index version, tool, arguments), so the same call always returns the same result.
