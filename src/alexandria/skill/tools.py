"""Tool definitions: one source for the OpenAI-style function schemas (vLLM tool calling / any OpenAI client),
the agent harness's JSON action schema, the MCP server and SKILL.md."""
from __future__ import annotations

import json

FIELDS = ["any", "claims", "key_results", "methods", "executive_summary", "limitations"]
PAPER_FIELDS = ["executive_summary", "research_question_hypothesis", "methodological_details", "key_results",
                "interpretation_implications", "contradictions_limitations", "claims"]

TOOLS = {
    "search": {
        "description": "Search structured summaries of scientific papers. Returns the best-matching summary "
                       "passages with paper ids. Use `field` to search one part of the summaries.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords or a question", "maxLength": 300},
            "field": {"type": "string", "enum": FIELDS, "description": "which summary field to search"},
            "k": {"type": "integer", "minimum": 1, "maximum": 10, "description": "number of results"}},
            "required": ["query"], "additionalProperties": False}},
    "get_paper": {
        "description": "Read selected summary fields of one paper by its id (from a search result).",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "string", "maxLength": 300},
            "fields": {"type": "array", "items": {"type": "string", "enum": PAPER_FIELDS}, "maxItems": 4}},
            "required": ["id"], "additionalProperties": False}},
    "search_fulltext": {
        "description": "Search the full text of open-access scientific papers. Returns matching text chunks with "
                       "paper ids.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "maxLength": 300},
            "k": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"], "additionalProperties": False}},
    "web_search": {
        "description": "Search the web. Returns titles, URLs and snippets.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "maxLength": 300}},
                       "required": ["query"], "additionalProperties": False}},
    "fetch": {
        "description": "Fetch the text of a web page by URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string", "maxLength": 500}},
                       "required": ["url"], "additionalProperties": False}},
}


def openai_tools(names: list[str]) -> list[dict]:
    """OpenAI / vLLM `tools=[...]` format."""
    return [{"type": "function", "function": {"name": n, **TOOLS[n]}} for n in names]


def tool_text(names: list[str]) -> str:
    """Tool descriptions for the prompt (the only part of the agent prompt that differs between arms)."""
    return "\n".join(json.dumps({"name": n, **TOOLS[n]}, ensure_ascii=False) for n in names)


def action_schema(tools: list[str], answer: dict, allow_tools: bool = True) -> dict:
    """JSON schema of one agent step: a thought, then either a tool call with valid arguments or the final answer.
    Constrained decoding with this schema makes every step well-formed (tool-call formatting is not a confound)."""
    options = [{"type": "object", "properties": {
        "thought": {"type": "string", "maxLength": 600},
        "action": {"const": "answer"},
        "arguments": {"type": "object", "properties": {"answer": answer}, "required": ["answer"],
                      "additionalProperties": False}},
        "required": ["thought", "action", "arguments"], "additionalProperties": False}]
    if allow_tools:
        for n in tools:
            options.append({"type": "object", "properties": {
                "thought": {"type": "string", "maxLength": 600},
                "action": {"const": n},
                "arguments": TOOLS[n]["parameters"]},
                "required": ["thought", "action", "arguments"], "additionalProperties": False})
    return options[0] if len(options) == 1 else {"anyOf": options}
