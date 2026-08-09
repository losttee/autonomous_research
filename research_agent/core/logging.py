"""Structured JSON logging; one line per pipeline step.

Each line carries step_id, step_type, tokens, latency_ms, cost_usd; the
monitoring dashboard aggregates from these.

Usage:
    from research_agent.core.logging import get_logger, log_step
    log = get_logger("planner")
    log_step(log, step_type="plan", step_id="plan_abc", tokens=1200,
             latency_ms=850, cost_usd=0.02, extra={"sub_tasks": 3})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from research_agent.core.config import get_settings


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON line. Custom fields live in record.extra_fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "component": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    settings = get_settings()
    if settings.log_json:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = logging.getLogger("research_agent")
    root.setLevel(settings.log_level.upper())
    root.handlers.clear()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    # File sink for the monitoring dashboard. A bad path must not stop the
    # pipeline, so fall back to stdout only.
    if settings.log_file:
        try:
            path = Path(settings.log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError:
            pass

    root.propagate = False
    _configured = True


def get_logger(component: str) -> logging.Logger:
    """Get a logger for one component (planner, executor, verifier, ...)."""
    _configure_root()
    return logging.getLogger(f"research_agent.{component}")


def log_step(
    logger: logging.Logger,
    *,
    step_type: str,
    step_id: str,
    tokens: int = 0,
    latency_ms: int = 0,
    cost_usd: float = 0.0,
    msg: str = "",
    extra: Optional[dict[str, Any]] = None,
    level: int = logging.INFO,
) -> None:
    """Log one pipeline step with standardized fields for measurement."""
    fields: dict[str, Any] = {
        "step_type": step_type,
        "step_id": step_id,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "cost_usd": round(cost_usd, 6),
    }
    if extra:
        fields.update(extra)
    logger.log(level, msg or step_type, extra={"extra_fields": fields})
