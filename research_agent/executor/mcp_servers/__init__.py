"""In-process tools (calculator, documents) implementing the same
search(query, max_results) -> list[SourceRef] interface as the web-search
backends.
"""

from research_agent.executor.mcp_servers.calculator import (
    CalcError,
    CalculatorTool,
    safe_eval,
)
from research_agent.executor.mcp_servers.documents import DocumentTool

__all__ = ["CalcError", "CalculatorTool", "DocumentTool", "safe_eval"]
