"""Tool servers behind the executor's fixed SearchTool interface.

Named mcp_servers per the roadmap; these are in-process tool servers (exact
calculator, internal documents), not networked Model Context Protocol
services — the executor only cares about the shared interface, and every
tool here implements the same `search(query, max_results) -> list[SourceRef]`
shape as the web-search backends.
"""

from research_agent.executor.mcp_servers.calculator import (
    CalcError,
    CalculatorTool,
    safe_eval,
)
from research_agent.executor.mcp_servers.documents import DocumentTool

__all__ = ["CalcError", "CalculatorTool", "DocumentTool", "safe_eval"]
