"""Advanced metrics calculator for guardrail benchmarking.

This module implements advanced evaluation metrics for benchmarking guardrail performance
across different models.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from .types import SampleResult

logger = logging.getLogger(__name__)


class BenchmarkMetricsCalculator:
    """Calculates advanced benchmarking metrics for guardrail evaluation."""

    def calculate_advanced_metrics(self, results: list[SampleResult], guardrail_name: str, guardrail_config: dict | None = None) -> dict[str, float]:
        """Calculate advanced metrics for a specific guardrail.

        Args:
            results: List of evaluation results
            guardrail_name: Name of the guardrail to analyze
            guardrail_config: Guardrail configuration to check for confidence thresholds

        Returns:
            Dictionary containing advanced metrics, or empty dict if not applicable
        """
        if not guardrail_config or "confidence_threshold" not in guardrail_config:
            return {}

        if not results:
            raise ValueError("Cannot calculate metrics for empty results list")

        y_true, y_scores = self._extract_labels_and_scores(results, guardrail_name)

        if not y_true:
            raise ValueError(f"No valid data found for guardrail '{guardrail_name}'")

        return self._calculate_metrics(y_true, y_scores)

    def _extract_labels_and_scores(self, results: list[SampleResult], guardrail_name: str) -> tuple[list[int], list[float]]:
        """Extract true labels and confidence scores for a guardrail."""
        y_true = []
        y_scores = []

        for result in results:
            if guardrail_name not in result.expected_triggers:
                logger.warning("Guardrail '%s' not found in expected_triggers for sample %s", guardrail_name, result.id)
                continue

            expected = result.expected_triggers[guardrail_name]
            y_true.append(1 if expected else 0)

            # Get confidence score from details, fallback to binary
            confidence = self._get_confidence_score(result, guardrail_name)
            y_scores.append(confidence)

        return y_true, y_scores

    def _get_confidence_score(self, result: SampleResult, guardrail_name: str) -> float:
        """Get confidence score for a guardrail result."""
        if guardrail_name in result.details:
            guardrail_details = result.details[guardrail_name]
            if isinstance(guardrail_details, dict) and "confidence" in guardrail_details:
                return float(guardrail_details["confidence"])

        # Fallback to binary: 1.0 if triggered, 0.0 if not
        actual = result.triggered.get(guardrail_name, False)
        return 1.0 if actual else 0.0

    def _calculate_metrics(self, y_true: list[int], y_scores: list[float]) -> dict[str, float]:
        """Calculate advanced metrics from labels and scores."""
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)

        metrics = {}

        # Calculate ROC AUC
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
        except ValueError as e:
            logger.warning("Could not calculate ROC AUC: %s", e)
            metrics["roc_auc"] = float("nan")

        # Calculate precision at different recall thresholds
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            metrics["prec_at_r80"] = self._precision_at_recall(precision, recall, 0.80)
            metrics["prec_at_r90"] = self._precision_at_recall(precision, recall, 0.90)
            metrics["prec_at_r95"] = self._precision_at_recall(precision, recall, 0.95)
        except Exception as e:
            logger.warning("Could not calculate precision at recall thresholds: %s", e)
            metrics.update({"prec_at_r80": float("nan"), "prec_at_r90": float("nan"), "prec_at_r95": float("nan")})

        # Calculate recall at FPR = 0.01
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            metrics["recall_at_fpr01"] = self._recall_at_fpr(fpr, tpr, 0.01)
        except Exception as e:
            logger.warning("Could not calculate recall at FPR=0.01: %s", e)
            metrics["recall_at_fpr01"] = float("nan")

        return metrics

    def _precision_at_recall(self, precision: np.ndarray, recall: np.ndarray, target_recall: float) -> float:
        """Find precision at a specific recall threshold."""
        valid_indices = np.where(recall >= target_recall)[0]

        if len(valid_indices) == 0:
            return 0.0

        best_idx = valid_indices[np.argmax(precision[valid_indices])]
        return float(precision[best_idx])

    def _recall_at_fpr(self, fpr: np.ndarray, tpr: np.ndarray, target_fpr: float) -> float:
        """Find recall (TPR) at a specific false positive rate threshold."""
        valid_indices = np.where(fpr <= target_fpr)[0]

        if len(valid_indices) == 0:
            return 0.0

        best_idx = valid_indices[np.argmax(tpr[valid_indices])]
        return float(tpr[best_idx])

    def calculate_all_guardrail_metrics(self, results: list[SampleResult]) -> dict[str, dict[str, float]]:
        """Calculate advanced metrics for all guardrails in the results."""
        if not results:
            return {}

        guardrail_names = set()
        for result in results:
            guardrail_names.update(result.expected_triggers.keys())

        metrics = {}
        for guardrail_name in guardrail_names:
            try:
                guardrail_metrics = self.calculate_advanced_metrics(results, guardrail_name)
                metrics[guardrail_name] = guardrail_metrics
            except Exception as e:
                logger.error("Failed to calculate metrics for guardrail '%s': %s", guardrail_name, e)
                metrics[guardrail_name] = {
                    "roc_auc": float("nan"),
                    "prec_at_r80": float("nan"),
                    "prec_at_r90": float("nan"),
                    "prec_at_r95": float("nan"),
                    "recall_at_fpr01": float("nan"),
                }

        return metrics
