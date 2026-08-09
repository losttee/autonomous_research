"""FastAPI service exposing the research pipeline over HTTP.

Run:

    .venv\\Scripts\\uvicorn.exe research_agent.api.main:app --reload

Then:
    curl -X POST http://127.0.0.1:8000/research \\
         -H "Content-Type: application/json" \\
         -d '{"question": "Compare plans A and B by premium"}'
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from monitoring.aggregate import aggregate
from research_agent.core.config import get_settings
from research_agent.core.contracts import FinalReport
from research_agent.core.pipeline import run_research
from research_agent.guardrail.cost_tracker import CostTracker

app = FastAPI(title="Autonomous Research & Decision Agent", version="0.2.0")

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/monitoring", include_in_schema=False)
def monitoring_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "monitoring.html")


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, description="The research question to answer")


class ResearchResponse(BaseModel):
    report: FinalReport
    cost_usd: float
    llm_calls: int
    tool_calls: int
    elapsed_sec: float


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    """Aggregated cost/latency from the structured log file (see monitoring/)."""
    return aggregate(get_settings().log_file)


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    tracker = CostTracker()
    report = run_research(req.question, tracker=tracker)
    snap = tracker.snapshot()
    return ResearchResponse(
        report=report,
        cost_usd=snap.cost_usd,
        llm_calls=snap.llm_calls,
        tool_calls=snap.tool_calls,
        elapsed_sec=snap.elapsed_sec,
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    """One Server-Sent Events frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/research/stream")
def research_stream(req: ResearchRequest) -> StreamingResponse:
    """Like /research, but pushes `progress` events over SSE while the run
    works, ending with one terminal event: `result` (same payload as
    /research) or `error`.
    """
    events: "queue.Queue[Optional[tuple[str, dict[str, Any]]]]" = queue.Queue()

    def on_progress(step: str, msg: str, **extra: Any) -> None:
        events.put(("progress", {"step": step, "msg": msg, **extra}))

    def worker() -> None:
        tracker = CostTracker()
        try:
            report = run_research(req.question, tracker=tracker, on_progress=on_progress)
            snap = tracker.snapshot()
            payload = ResearchResponse(
                report=report,
                cost_usd=snap.cost_usd,
                llm_calls=snap.llm_calls,
                tool_calls=snap.tool_calls,
                elapsed_sec=snap.elapsed_sec,
            ).model_dump(mode="json")
            events.put(("result", payload))
        except Exception as exc:
            events.put(("error", {"message": str(exc)}))
        finally:
            events.put(None)  # sentinel: closes the stream

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            item = events.get()
            if item is None:
                break
            event, data = item
            yield _sse(event, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
