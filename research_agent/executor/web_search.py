"""Web search backends behind one SearchTool interface.

TavilySearchTool is the real API; StubSearchTool is the offline stand-in used
when no key is configured, so the pipeline runs without network.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from research_agent.core.config import Settings, get_settings
from research_agent.core.contracts import SourceRef, SourceType
from research_agent.core.logging import get_logger

_log = get_logger("executor.web_search")

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class SearchTool(Protocol):
    """Fixed interface for any web-search backend."""

    name: str

    def search(self, query: str, max_results: int = 5) -> list[SourceRef]:
        ...


class StubSearchTool:
    """Offline deterministic search for tests and no-key runs."""

    name = "stub"

    def search(self, query: str, max_results: int = 5) -> list[SourceRef]:
        n = max(1, min(max_results, 3))
        sources: list[SourceRef] = []
        for i in range(n):
            sources.append(
                SourceRef(
                    type=SourceType.WEB,
                    title=f"[stub] Result {i + 1} for: {query[:60]}",
                    url=f"https://example.com/stub/{i + 1}",
                    snippet=(
                        f"Stub snippet {i + 1} relevant to '{query[:80]}'. "
                        "Replace with a real search backend by setting TAVILY_API_KEY."
                    ),
                    reliability=0.5,
                )
            )
        _log.info(
            "stub search",
            extra={"extra_fields": {"query": query[:80], "results": len(sources)}},
        )
        return sources


class TavilySearchTool:
    """Real web search via the Tavily API."""

    name = "tavily"

    def __init__(self, api_key: str, timeout_sec: float = 20.0) -> None:
        self._api_key = api_key
        self._timeout = timeout_sec

    def search(self, query: str, max_results: int = 5) -> list[SourceRef]:
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }
        try:
            resp = httpx.post(TAVILY_ENDPOINT, json=payload, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            _log.warning(
                "tavily search failed",
                extra={"extra_fields": {"query": query[:80], "error": str(exc)}},
            )
            return []

        sources: list[SourceRef] = []
        for item in data.get("results", [])[:max_results]:
            sources.append(
                SourceRef(
                    type=SourceType.WEB,
                    title=item.get("title", ""),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:1000],
                    reliability=item.get("score"),
                )
            )
        _log.info(
            "tavily search",
            extra={"extra_fields": {"query": query[:80], "results": len(sources)}},
        )
        return sources


def get_search_tool(settings: Settings | None = None) -> SearchTool:
    """Return the real tool if an API key is configured, else the offline stub."""
    settings = settings or get_settings()
    if settings.tavily_api_key:
        return TavilySearchTool(settings.tavily_api_key)
    return StubSearchTool()
