"""Calculator tool — exact arithmetic as a grounded source.

Financial/math figures must never come from an LLM guessing: the model only
*extracts* the expression from the question (pattern recognition), the math
itself runs through a restricted AST evaluator, and the result is returned as
a SourceRef(type=CALCULATOR) with reliability 1.0 — cited like any other
source by the executor/verifier/synthesizer.

Degradation (same rule as every layer): if the extraction LLM is unavailable,
a regex fallback tries to find a plain arithmetic expression in the question;
if nothing computable exists, the tool returns [] and the sub-task degrades.
"""

from __future__ import annotations

import ast
import re
from typing import TYPE_CHECKING, Optional

from research_agent.core.contracts import SourceRef, SourceType
from research_agent.core.logging import get_logger

if TYPE_CHECKING:
    from research_agent.core.llm import LLMClient
    from research_agent.guardrail.cost_tracker import CostTracker

_log = get_logger("executor.mcp.calculator")


class CalcError(ValueError):
    """Raised when an expression is unsafe or cannot be evaluated."""


# Only pure arithmetic survives validation — no names, calls, attributes,
# subscripts, strings. Anything else is rejected before eval ever sees it.
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)

# 9**9**9 would try to materialize hundreds of millions of digits; exponents
# must be small literal numbers.
_MAX_EXPONENT = 64


def safe_eval(expression: str) -> float:
    """Evaluate a pure arithmetic expression safely via a whitelisted AST.

    Raises CalcError on any disallowed syntax, non-numeric constants, large
    exponents, division by zero, or overflow.
    """
    expr = expression.strip()
    if not expr:
        raise CalcError("empty expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"unparseable expression: {exc}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise CalcError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise CalcError("only numeric constants allowed")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            right = node.right
            ok = (
                isinstance(right, ast.Constant)
                and isinstance(right.value, (int, float))
                and abs(right.value) <= _MAX_EXPONENT
            )
            if not ok:
                raise CalcError("exponent must be a small literal number")

    try:
        result = eval(  # noqa: S307 — AST-whitelisted arithmetic only
            compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, {}
        )
    except ZeroDivisionError as exc:
        raise CalcError("division by zero") from exc
    except OverflowError as exc:
        raise CalcError("numeric overflow") from exc
    return float(result)


def _format(value: float) -> str:
    """Render results without float noise: 1024.0 -> '1024', else 10 sig figs."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.10g}"


_EXTRACT_SYSTEM = (
    "You extract ONE arithmetic expression from a question. You never compute "
    "results — you only translate words into math. Return STRICT JSON only."
)

_EXTRACT_TEMPLATE = """Question:
{question}

Return a JSON object of this exact shape:
{{"expression": "<Python arithmetic expression, or empty string if there is nothing to compute>"}}

Rules:
- Only digits, parentheses and the operators + - * / ** (no % sign).
- Convert percentages to decimals: "25%" -> 0.25.
- No variable names, no function calls, no units inside the expression.
- If the question has nothing to compute, return {{"expression": ""}}."""

# LLM-free fallback: the first run of digits/operators in the question.
_FALLBACK_RE = re.compile(r"\d[\d\.\s\+\-\*/\(\)]*\d")


def _fallback_expression(query: str) -> Optional[str]:
    match = _FALLBACK_RE.search(query)
    if not match:
        return None
    expr = re.sub(r"\s+", "", match.group(0))
    return expr or None


class CalculatorTool:
    """Exact-arithmetic tool behind the executor's fixed SearchTool interface."""

    name = "calculator"

    def __init__(
        self,
        llm: Optional["LLMClient"] = None,
        tracker: Optional["CostTracker"] = None,
    ) -> None:
        self._llm = llm
        self._tracker = tracker

    def _extract_expression(self, query: str) -> Optional[str]:
        from research_agent.core.config import get_settings
        from research_agent.core.llm import get_llm_client

        client = self._llm or get_llm_client()
        try:
            data = client.complete_json(
                _EXTRACT_TEMPLATE.format(question=query),
                model=get_settings().worker_model,
                system=_EXTRACT_SYSTEM,
                max_tokens=120,
                temperature=0.0,
                tracker=self._tracker,
            )
        except Exception as exc:  # defensive: tool errors never crash a run
            _log.warning(
                "calculator extraction fell back to regex",
                extra={"extra_fields": {"error": str(exc)}},
            )
            return _fallback_expression(query)
        if isinstance(data, dict):
            expr = str(data.get("expression", "")).strip()
            if expr:
                return expr
        return _fallback_expression(query)

    def search(self, query: str, max_results: int = 1) -> list[SourceRef]:
        """Compute the expression found in the query; one source or nothing."""
        expr = self._extract_expression(query)
        if not expr:
            _log.info(
                "calculator found nothing to compute",
                extra={"extra_fields": {"query": query[:80]}},
            )
            return []
        try:
            value = safe_eval(expr)
        except CalcError as exc:
            _log.warning(
                "calculator rejected expression",
                extra={"extra_fields": {"expression": expr, "error": str(exc)}},
            )
            return []
        rendered = f"{expr} = {_format(value)}"
        _log.info(
            "calculator computed",
            extra={"extra_fields": {"expression": expr, "result": _format(value)}},
        )
        return [
            SourceRef(
                type=SourceType.CALCULATOR,
                title=f"Calculation: {expr}",
                snippet=rendered,
                reliability=1.0,  # exact arithmetic carries no doubt
            )
        ]
