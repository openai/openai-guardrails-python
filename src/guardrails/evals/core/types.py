"""Core types and protocols for guardrail evaluation.

This module defines the core data models and protocols used throughout the guardrail evaluation framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openai import AsyncOpenAI

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None  # type: ignore

from pydantic import BaseModel


class Sample(BaseModel):
    """A single evaluation sample.

    Attributes:
        id: Unique identifier for the sample.
        data: The text or data to be evaluated.
        expected_triggers: Mapping of guardrail names to expected trigger status.
    """

    id: str
    data: str
    expected_triggers: dict[str, bool]


class SampleResult(BaseModel):
    """A single sample result.

    Attributes:
        id: Unique identifier for the sample.
        expected_triggers: Mapping of guardrail names to expected trigger status.
        triggered: Mapping of guardrail names to actual trigger status.
        details: Additional details for each guardrail.
    """

    id: str
    expected_triggers: dict[str, bool]
    triggered: dict[str, bool]
    details: dict[str, Any]


class GuardrailMetrics(BaseModel):
    """Guardrail evaluation metrics.

    Attributes:
        true_positives: Number of true positives.
        false_positives: Number of false positives.
        false_negatives: Number of false negatives.
        true_negatives: Number of true negatives.
        total_samples: Total number of samples evaluated.
        precision: Precision score.
        recall: Recall score.
        f1_score: F1 score.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    total_samples: int
    precision: float
    recall: float
    f1_score: float


@dataclass(frozen=True, slots=True)
class Context:
    """Evaluation context with LLM client.

    Supports OpenAI, Azure OpenAI, and OpenAI-compatible APIs.

    Attributes:
        guardrail_llm: Asynchronous OpenAI or Azure OpenAI client for LLM-based guardrails.
    """

    guardrail_llm: AsyncOpenAI | AsyncAzureOpenAI  # type: ignore


class DatasetLoader(Protocol):
    """Protocol for dataset loading and validation."""

    def load(self, path: Path) -> list[Sample]:
        """Load and validate dataset from path."""
        ...


class RunEngine(Protocol):
    """Protocol for running guardrail evaluations."""

    async def run(self, context: Context, samples: list[Sample], batch_size: int) -> list[SampleResult]:
        """Run evaluations on samples."""
        ...


class MetricsCalculator(Protocol):
    """Protocol for calculating evaluation metrics."""

    def calculate(self, results: list[SampleResult]) -> dict[str, GuardrailMetrics]:
        """Calculate metrics from results."""
        ...


class ResultsReporter(Protocol):
    """Protocol for reporting evaluation results."""

    def save(
        self,
        results: list[SampleResult],
        metrics: dict[str, GuardrailMetrics],
        output_dir: Path,
    ) -> None:
        """Save results and metrics to output directory."""
        ...
