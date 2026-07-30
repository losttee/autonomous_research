"""Monitoring layer — aggregates the pipeline's structured logs into numbers.

The pipeline writes one JSON line per step to LOG_FILE (see core/logging.py).
This module turns that stream into per-step-type rollups and a recent-runs
list, consumed by the /metrics endpoint and the /monitoring dashboard page.
"""

from monitoring.aggregate import aggregate

__all__ = ["aggregate"]
