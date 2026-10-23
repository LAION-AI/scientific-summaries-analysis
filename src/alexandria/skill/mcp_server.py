"""MCP server exposing the Alexandria skill tools over stdio (or streamable HTTP).

    alexandria skill mcp --profile debug [--view summary] [--transport stdio|streamable-http]

Register it with any MCP client (e.g. `claude mcp add alexandria -- alexandria skill mcp`). Needs the `mcp` package
(present in the Alex-vllm env; mcp 2.x `MCPServer`, mcp 1.x `FastMCP`)."""
from __future__ import annotations

from .backend import Backend
from .tools import TOOLS


def build_server(backend: Backend, view: str = "summary"):
    try:
        from mcp.server.mcpserver import MCPServer as Server                     # mcp >= 2
    except ImportError:
        from mcp.server.fastmcp import FastMCP as Server                         # mcp 1.x
    from pathlib import Path
    srv = Server("alexandria", instructions=(Path(__file__).with_name("SKILL.md").read_text()))

    @srv.tool(name="search", description=TOOLS["search"]["description"])
    def search(query: str, field: str = "any", k: int = 5) -> str:
        return backend.call("search", {"query": query, "field": field, "k": k}, view=view)[0]

    @srv.tool(name="get_paper", description=TOOLS["get_paper"]["description"])
    def get_paper(id: str, fields: list[str] | None = None) -> str:            # noqa: A002  (tool argument name)
        return backend.call("get_paper", {"id": id, "fields": fields}, view=view)[0]

    @srv.tool(name="search_fulltext", description=TOOLS["search_fulltext"]["description"])
    def search_fulltext(query: str, k: int = 5) -> str:
        return backend.call("search_fulltext", {"query": query, "k": k}, view="fulltext")[0]

    return srv
