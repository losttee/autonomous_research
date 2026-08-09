"""Document tool: internal files as INTERNAL_RAG sources.

Searches DOCUMENT_ROOT for text files (.md/.txt/.json/.csv/...) by keyword
overlap and returns the best matches as SourceRefs. Only files under the
resolved root are readable; a missing root returns []. PDF needs a parsing
library and is not supported yet.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from research_agent.core.contracts import SourceRef, SourceType
from research_agent.core.logging import get_logger

_log = get_logger("executor.mcp.documents")

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_TEXT_EXTENSIONS = {".md", ".txt", ".json", ".csv", ".tsv", ".log", ".rst"}

# Read cap per file: internal notes are small; a giant file must not stall a run.
_MAX_READ_CHARS = 200_000
_SNIPPET_CHARS = 500

# Internal docs: trustworthy but possibly stale.
_INTERNAL_RELIABILITY = 0.7


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _best_snippet(text: str, query_tokens: set[str]) -> str:
    """Every paragraph overlapping the query joined together, else the head."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    relevant = [p for p in paragraphs[:200] if query_tokens & _tokens(p)]
    source = "\n".join(relevant) if relevant else text
    return source[:_SNIPPET_CHARS].strip()


class DocumentTool:
    """Keyword search over a local document folder, behind the fixed interface."""

    name = "documents"

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            from research_agent.core.config import get_settings

            root = get_settings().document_root
        self._root = Path(root)

    def search(self, query: str, max_results: int = 5) -> list[SourceRef]:
        root = self._root
        if not root.is_dir():
            _log.info(
                "document root missing",
                extra={"extra_fields": {"root": str(root)}},
            )
            return []
        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        try:
            root_resolved = root.resolve()
        except OSError:
            return []

        scored: list[tuple[float, Path, str]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _TEXT_EXTENSIONS:
                continue
            try:
                if root_resolved not in path.resolve().parents:
                    continue  # stay inside the root
                text = path.read_text(encoding="utf-8", errors="ignore")[:_MAX_READ_CHARS]
            except OSError:
                continue
            file_tokens = _tokens(path.name) | _tokens(text[:5000])
            overlap = query_tokens & file_tokens
            if not overlap:
                continue
            score = len(overlap) / len(query_tokens)
            scored.append((score, path, _best_snippet(text, query_tokens)))

        scored.sort(key=lambda item: item[0], reverse=True)
        sources: list[SourceRef] = []
        for _, path, snippet in scored[:max_results]:
            sources.append(
                SourceRef(
                    type=SourceType.INTERNAL_RAG,
                    title=path.name,
                    url=path.relative_to(root).as_posix(),
                    snippet=snippet,
                    reliability=_INTERNAL_RELIABILITY,
                )
            )
        if sources:
            _log.info(
                "documents matched",
                extra={"extra_fields": {"query": query[:80], "results": len(sources)}},
            )
        return sources
