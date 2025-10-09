"""Metrics calculator for guardrail evaluation.

This module implements precision, recall, and F1-score calculation for guardrail evaluation results.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .types import GuardrailMetrics, MetricsCalculator, SampleResult

logger = logging.getLogger(__name__)


class GuardrailMetricsCalculator(MetricsCalculator):
    """Calculates evaluation metrics from results."""

    def calculate(self, results: Sequence[SampleResult]) -> dict[str, GuardrailMetrics]:
        """Calculate precision, recall, and F1 score for each guardrail.

        Args:
            results: Sequence of evaluation results

        Returns:
            Dictionary mapping guardrail names to their metrics

        Raises:
            ValueError: If results list is empty
        """
        if not results:
            raise ValueError("Cannot calculate metrics for empty results list")

        guardrail_names = results[0].triggered.keys()
        metrics: dict[str, GuardrailMetrics] = {}

        for name in guardrail_names:
            metrics[name] = self._calculate_guardrail_metrics(results, name)

        return metrics

    def _calculate_guardrail_metrics(self, results: Sequence[SampleResult], name: str) -> GuardrailMetrics:
        """Calculate metrics for a specific guardrail."""
        true_positives = sum(1 for r in results if r.expected_triggers.get(name) and r.triggered.get(name))
        false_positives = sum(1 for r in results if not r.expected_triggers.get(name) and r.triggered.get(name))
        false_negatives = sum(1 for r in results if r.expected_triggers.get(name) and not r.triggered.get(name))
        true_negatives = sum(1 for r in results if not r.expected_triggers.get(name) and not r.triggered.get(name))

        total = true_positives + false_positives + false_negatives + true_negatives
        if total != len(results):
            logger.error(
                "Metrics sum mismatch for %s: %d != %d",
                name,
                total,
                len(results),
            )
            raise ValueError(f"Metrics sum mismatch for {name}")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return GuardrailMetrics(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives,
            total_samples=total,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
        )
