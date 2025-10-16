"""Evaluation tools and utilities for guardrails.

This package contains tools for evaluating guardrails models and configurations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from guardrails.evals.core import (
    AsyncRunEngine,
    BenchmarkMetricsCalculator,
    BenchmarkReporter,
    BenchmarkVisualizer,
    GuardrailMetricsCalculator,
    JsonlDatasetLoader,
    JsonResultsReporter,
    LatencyTester,
    validate_dataset,
)

if TYPE_CHECKING:
    from guardrails.evals.guardrail_evals import GuardrailEval

__all__ = [
    "GuardrailEval",
    "AsyncRunEngine",
    "BenchmarkMetricsCalculator",
    "BenchmarkReporter",
    "BenchmarkVisualizer",
    "GuardrailMetricsCalculator",
    "JsonResultsReporter",
    "JsonlDatasetLoader",
    "LatencyTester",
    "validate_dataset",
]


def __getattr__(name: str) -> Any:
    if name == "GuardrailEval":
        from guardrails.evals.guardrail_evals import GuardrailEval as _GuardrailEval

        return _GuardrailEval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
