# Autonomous Research & Decision Agent

A multi-layer deep-research agent. Given a question, it autonomously **plans** the work, **gathers** evidence from multiple sources, **verifies** each claim against those sources, and **synthesizes** a cited report with explicit confidence levels. Every assertion is traceable back to a source — no source, no claim.

## Overview

The system is organized as a pipeline of specialized layers. Each layer has a single responsibility and communicates through shared data contracts defined in `research_agent/core/contracts.py`, so every piece of data carries traceability metadata (source id, retrieval time, confidence) as it flows through.

```
Question
   │
   ▼
Planner       Decompose the question into sub-tasks, with a dependency graph
   │           so independent sub-tasks can run in parallel.
   ▼
Executor      Run each sub-task with tools (web search, ...) and return
   │           distilled claims plus their supporting sources.
   ▼
Verifier      Check each claim for grounding, attach a confidence score,
   │           and surface contradictions between sources.
   ▼
Synthesizer   Produce the final report: recommendation, cited analysis,
   │           and an explicit account of what remains uncertain.
   ▼
FinalReport (JSON)
```

Two concerns cut across the whole pipeline:

- **Guardrail** (`guardrail/cost_tracker.py`) — enforces hard per-request limits on LLM calls, tool calls, tokens, spend, and wall-clock time to prevent runaway cost.
- **Memory** (`memory/`) — working memory and vector store, used by later stages for retrieval and recall.

The planner assigns one tool per sub-task (`web`, `calculator`, or `documents`)
and the executor dispatches on that hint:

- **Calculator** — the LLM only extracts the arithmetic expression; the math runs
  through a restricted AST evaluator, so computed numbers are exact and cited as
  `CALCULATOR` sources (reliability 1.0).
- **Documents** — keyword search over `DOCUMENT_ROOT` (markdown/text/JSON/CSV),
  surfaced as `INTERNAL_RAG` sources so internal files are cited like web pages.

## Project layout

```
research_agent/
├── core/          Data contracts, configuration, pipeline, LLM client, logging
├── planner/       Decomposes a question into sub-tasks
├── executor/      Runs sub-tasks; tooling in executor/mcp_servers/
│                  (web search, exact calculator, internal documents)
├── verifier/      Grounding and claim verification
├── synthesizer/   Generates the final report
├── guardrail/     Cost tracking and enforcement
├── memory/        Working memory / vector database
└── api/           FastAPI service and static frontend
tests/             Test suites, organized incrementally
```

## Installation

Requires Python 3.11 or newer.

```powershell
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Copy the configuration template and fill in real values:

```powershell
copy .env.example .env
```

Set `LLM_API_KEY` to a key for your OpenAI-compatible endpoint. See `research_agent/core/config.py` for the meaning of every variable.

> `.env` holds secrets and is git-ignored. Never commit it.

## Running

Start the API together with the web interface:

```powershell
.venv\Scripts\python.exe -m uvicorn research_agent.api.main:app --reload --port 8000
```

Open **http://127.0.0.1:8000/** in a browser, enter a question, and click **Nghiên cứu** (or press `Ctrl` + `Enter`).

Call the API directly:

```powershell
curl -X POST http://127.0.0.1:8000/research `
     -H "Content-Type: application/json" `
     -d '{\"question\": \"Compare plans A and B by premium\"}'
```

### Endpoints

| Method | Path              | Description                             |
|--------|-------------------|-----------------------------------------|
| GET    | `/`               | Web interface                           |
| GET    | `/monitoring`     | Cost & latency dashboard                |
| GET    | `/health`         | Liveness check                          |
| GET    | `/metrics`        | Aggregated cost/latency as JSON         |
| POST   | `/research`       | Accept a question, return a `FinalReport` as JSON |
| POST   | `/research/stream`| Same, streaming SSE progress events while it runs |

## Configuration

Hard per-request limits (set in `.env`) keep cost bounded during experimentation:

| Variable              | Description                          |
|-----------------------|--------------------------------------|
| `MAX_LLM_CALLS`       | Maximum number of LLM invocations    |
| `MAX_TOOL_CALLS`      | Maximum number of tool invocations   |
| `MAX_TOKENS_BUDGET`   | Token budget                         |
| `MAX_USD_BUDGET`      | Spend budget in USD                  |
| `REQUEST_TIMEOUT_SEC` | Per-request timeout in seconds       |
| `MAX_REPLAN`          | Maximum number of re-planning rounds |
| `MODEL_PRICING`       | Optional price overrides, `model:in:out,...` USD/1M tokens |
| `LOG_FILE`            | JSON-lines log the monitoring dashboard reads |

Model tiers (`PLANNER_MODEL`, `WORKER_MODEL`, `VERIFIER_MODEL`) let you assign a stronger model to planning and cheaper models to workers and verification.

## Evaluation

`evaluation/` measures what the pipeline promises: citations that resolve,
grounded claims, honest uncertainty, and per-question cost — over a golden set
of factual, open-ended, and deliberately unanswerable questions.

```powershell
# Run the golden set through the REAL pipeline (real LLM cost; guardrails apply)
.venv\Scripts\python.exe -m evaluation.run_eval

# Fewer questions, skip the LLM grounding judge
.venv\Scripts\python.exe -m evaluation.run_eval --limit 3 --no-judge

# Compare two runs side by side
.venv\Scripts\python.exe -m evaluation.compare evaluation\results\a.json evaluation\results\b.json
```

Each run writes `evaluation/results/<timestamp>.json`. Keyword-recall and
honesty metrics are only meaningful with a real search backend
(`TAVILY_API_KEY`) — the offline stub cannot produce real facts.

## Monitoring

Every pipeline step writes one structured JSON line (`step_type`, `tokens`,
`latency_ms`, `cost_usd`) to `LOG_FILE` (`data/logs/pipeline.jsonl` by default).
`monitoring/aggregate.py` rolls that stream up; open **/monitoring** in the
browser for the dashboard (cost per layer, per-run history, budget-cap hits),
or fetch **/metrics** for the raw JSON.

The web UI streams progress over SSE on **/research/stream** — the question
page shows each stage (planning → gathering → verifying → synthesizing) live
instead of a blank wait, and falls back to the classic endpoint automatically.

## Testing

```powershell
.venv\Scripts\python.exe -m pytest
```

## License

All rights reserved by the project owner.
