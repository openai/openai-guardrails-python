"""Visualization module for guardrail benchmarking.

This module generates charts and graphs for benchmark results.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class BenchmarkVisualizer:
    """Generates visualizations for guardrail benchmark results."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the visualizer.

        Args:
            output_dir: Directory to save generated charts
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style and color palette
        plt.style.use("default")
        self.colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        sns.set_palette(self.colors)

    def create_all_visualizations(
        self,
        results_by_model: dict[str, list[Any]],
        metrics_by_model: dict[str, dict[str, float]],
        latency_results: dict[str, dict[str, Any]],
        guardrail_name: str,
        expected_triggers: dict[str, bool],
    ) -> list[Path]:
        """Create all visualizations for a benchmark run.

        Args:
            results_by_model: Dictionary mapping model names to their results
            metrics_by_model: Dictionary mapping model names to their metrics
            latency_results: Dictionary mapping model names to their latency data
            guardrail_name: Name of the guardrail being evaluated
            expected_triggers: Expected trigger values for each sample

        Returns:
            List of paths to saved visualization files
        """
        saved_files = []

        # Create ROC curves
        try:
            roc_file = self.create_roc_curves(results_by_model, guardrail_name, expected_triggers)
            saved_files.append(roc_file)
        except Exception as e:
            logger.error("Failed to create ROC curves: %s", e)

        # Create basic performance metrics chart
        try:
            basic_metrics = self._extract_basic_metrics(metrics_by_model)
            if basic_metrics:
                basic_file = self.create_basic_metrics_chart(basic_metrics, guardrail_name)
                saved_files.append(basic_file)
        except Exception as e:
            logger.error("Failed to create basic metrics chart: %s", e)

        # Create advanced performance metrics chart (only if advanced metrics exist)
        try:
            if any("prec_at_r80" in metrics for metrics in metrics_by_model.values()):
                advanced_file = self.create_advanced_metrics_chart(metrics_by_model, guardrail_name)
                saved_files.append(advanced_file)
        except Exception as e:
            logger.error("Failed to create advanced metrics chart: %s", e)

        # Create latency comparison chart
        try:
            latency_file = self.create_latency_comparison_chart(latency_results)
            saved_files.append(latency_file)
        except Exception as e:
            logger.error("Failed to create latency comparison chart: %s", e)

        return saved_files

    def create_roc_curves(self, results_by_model: dict[str, list[Any]], guardrail_name: str, expected_triggers: dict[str, bool]) -> Path:
        """Create ROC curves comparing models for a specific guardrail."""
        fig, ax = plt.subplots(figsize=(10, 8))

        for model_name, results in results_by_model.items():
            y_true, y_scores = self._extract_roc_data(results, guardrail_name)

            if not y_true:
                logger.warning("No valid data for model %s and guardrail %s", model_name, guardrail_name)
                continue

            try:
                from sklearn.metrics import roc_curve

                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = np.trapz(tpr, fpr)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
            except Exception as e:
                logger.error("Failed to calculate ROC curve for model %s: %s", model_name, e)

        # Add diagonal line and customize plot
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Classifier")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title(f"ROC Curves: {guardrail_name} Performance Across Models", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        # Save plot
        filename = f"{guardrail_name}_roc_curves.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("ROC curves saved to: %s", filepath)
        return filepath

    def _extract_roc_data(self, results: list[Any], guardrail_name: str) -> tuple[list[int], list[float]]:
        """Extract true labels and predictions for ROC curve."""
        y_true = []
        y_scores = []

        for result in results:
            if guardrail_name in result.expected_triggers:
                expected = result.expected_triggers[guardrail_name]
                actual = result.triggered.get(guardrail_name, False)

                y_true.append(1 if expected else 0)
                y_scores.append(1 if actual else 0)

        return y_true, y_scores

    def create_latency_comparison_chart(self, latency_results: dict[str, dict[str, Any]]) -> Path:
        """Create a chart comparing latency across models."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        models = list(latency_results.keys())
        metrics = ["P50", "P95"]
        x = np.arange(len(metrics))
        width = 0.8 / len(models)

        # Extract P50 and P95 values for each model
        for i, model in enumerate(models):
            ttft_p50 = self._safe_get_latency_value(latency_results[model], "ttft", "p50")
            ttft_p95 = self._safe_get_latency_value(latency_results[model], "ttft", "p95")
            ttc_p50 = self._safe_get_latency_value(latency_results[model], "ttc", "p50")
            ttc_p95 = self._safe_get_latency_value(latency_results[model], "ttc", "p95")

            offset = (i - len(models) / 2 + 0.5) * width

            # Time to First Token chart
            ax1.bar(x + offset, [ttft_p50, ttft_p95], width, label=model, alpha=0.8)

            # Time to Completion chart
            ax2.bar(x + offset, [ttc_p50, ttc_p95], width, label=model, alpha=0.8)

        # Setup charts
        for ax, title in [(ax1, "Time to First Token (TTFT)"), (ax2, "Time to Completion (TTC)")]:
            ax.set_xlabel("Metrics", fontsize=12)
            ax.set_ylabel("Latency (ms)", fontsize=12)
            ax.set_title(title, fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        # Save plot
        filename = "latency_comparison.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Latency comparison chart saved to: %s", filepath)
        return filepath

    def _safe_get_latency_value(self, latency_data: dict[str, Any], metric_type: str, percentile: str) -> float:
        """Safely extract latency value, returning 0 if not available."""
        if metric_type in latency_data and isinstance(latency_data[metric_type], dict):
            value = latency_data[metric_type].get(percentile, float("nan"))
            return 0 if np.isnan(value) else value
        return 0.0

    def _extract_basic_metrics(self, metrics_by_model: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Extract basic metrics from the full metrics."""
        basic_metrics = {}
        for model_name, metrics in metrics_by_model.items():
            basic_metrics[model_name] = {
                "roc_auc": metrics.get("roc_auc", float("nan")),
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "f1_score": metrics.get("f1_score", float("nan")),
            }
        return basic_metrics

    def create_basic_metrics_chart(self, metrics_by_model: dict[str, dict[str, float]], guardrail_name: str) -> Path:
        """Create a grouped bar chart comparing basic performance metrics across models."""
        metric_names = ["Precision", "Recall", "F1 Score"]
        metric_keys = ["precision", "recall", "f1_score"]

        models = list(metrics_by_model.keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped bars
        for i, model in enumerate(models):
            model_metrics = metrics_by_model[model]
            values = [model_metrics.get(key, float("nan")) for key in metric_keys]
            values = [0 if np.isnan(v) else v for v in values]

            bar_positions = x + i * width - (len(models) - 1) * width / 2
            bars = ax.bar(bar_positions, values, width, label=model, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

        # Customize plot
        ax.set_xlabel("Performance Metrics", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Basic Performance Metrics: {guardrail_name}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.legend(title="Models", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        # Save plot
        filename = f"{guardrail_name}_basic_metrics.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Basic metrics chart saved to %s", filepath)
        return filepath

    def create_advanced_metrics_chart(self, metrics_by_model: dict[str, dict[str, float]], guardrail_name: str) -> Path:
        """Create a grouped bar chart comparing advanced performance metrics across models."""
        metric_names = ["ROC AUC", "Prec@R=0.80", "Prec@R=0.90", "Prec@R=0.95", "Recall@FPR=0.01"]
        metric_keys = ["roc_auc", "prec_at_r80", "prec_at_r90", "prec_at_r95", "recall_at_fpr01"]

        models = list(metrics_by_model.keys())
        x = np.arange(len(metric_names))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(14, 8))

        # Create grouped bars
        for i, model in enumerate(models):
            model_metrics = metrics_by_model[model]
            values = [model_metrics.get(key, float("nan")) for key in metric_keys]
            values = [0 if np.isnan(v) else v for v in values]

            bar_positions = x + i * width - (len(models) - 1) * width / 2
            bars = ax.bar(bar_positions, values, width, label=model, alpha=0.8)

            # Add value labels on bars
            for bar, value in zip(bars, values, strict=False):
                if value > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

        # Customize plot
        ax.set_xlabel("Performance Metrics", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"Advanced Performance Metrics: {guardrail_name}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45, ha="right")
        ax.legend(title="Models", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.1)

        plt.tight_layout()

        # Save plot
        filename = f"{guardrail_name}_advanced_metrics.png"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Advanced metrics chart saved to %s", filepath)
        return filepath
