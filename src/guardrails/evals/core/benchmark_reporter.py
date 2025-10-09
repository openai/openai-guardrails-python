"""Benchmark results reporter for guardrail evaluation.

This module handles saving benchmark results in a specialized format with analysis
folders containing visualizations and detailed metrics.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .types import SampleResult

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """Reports benchmark results with specialized output format."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the benchmark reporter.

        Args:
            output_dir: Base directory for benchmark results
        """
        self.output_dir = output_dir

    def save_benchmark_results(
        self,
        results_by_model: dict[str, list[SampleResult]],
        metrics_by_model: dict[str, dict[str, float]],
        latency_results: dict[str, dict[str, Any]],
        guardrail_name: str,
        dataset_size: int,
        latency_iterations: int,
    ) -> Path:
        """Save benchmark results in organized folder structure.

        Args:
            results_by_model: Dictionary mapping model names to their results
            metrics_by_model: Dictionary mapping model names to their metrics
            latency_results: Dictionary mapping model names to their latency data
            guardrail_name: Name of the guardrail being benchmarked
            dataset_size: Number of samples in the dataset
            latency_iterations: Number of iterations used for latency testing

        Returns:
            Path to the benchmark results directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        benchmark_dir = self.output_dir / f"benchmark_{guardrail_name}_{timestamp}"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        results_dir = benchmark_dir / "results"
        graphs_dir = benchmark_dir / "graphs"
        results_dir.mkdir(exist_ok=True)
        graphs_dir.mkdir(exist_ok=True)

        try:
            # Save per-model results
            for model_name, results in results_by_model.items():
                model_results_file = results_dir / f"eval_results_{guardrail_name}_{model_name}.jsonl"
                self._save_results_jsonl(results, model_results_file)
                logger.info("Model %s results saved to %s", model_name, model_results_file)

            # Save combined data
            self._save_metrics_json(metrics_by_model, results_dir / "performance_metrics.json")
            self._save_latency_json(latency_results, results_dir / "latency_results.json")

            # Save summary files
            summary_file = benchmark_dir / "benchmark_summary.txt"
            self._save_benchmark_summary(
                summary_file, guardrail_name, results_by_model, metrics_by_model, latency_results, dataset_size, latency_iterations
            )

            self._save_summary_tables(benchmark_dir, metrics_by_model, latency_results)

        except Exception as e:
            logger.error("Failed to save benchmark results: %s", e)
            raise

        logger.info("Benchmark results saved to: %s", benchmark_dir)
        return benchmark_dir

    def _create_performance_table(self, metrics_by_model: dict[str, dict[str, float]]) -> pd.DataFrame:
        """Create a performance metrics table."""
        if not metrics_by_model:
            return pd.DataFrame()

        metric_keys = ["precision", "recall", "f1_score", "roc_auc"]
        metric_names = ["Precision", "Recall", "F1 Score", "ROC AUC"]

        table_data = []
        for model_name, model_metrics in metrics_by_model.items():
            row = {"Model": model_name}
            for key, display_name in zip(metric_keys, metric_names, strict=False):
                value = model_metrics.get(key, float("nan"))
                row[display_name] = "N/A" if pd.isna(value) else f"{value:.4f}"
            table_data.append(row)

        return pd.DataFrame(table_data)

    def _create_latency_table(self, latency_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
        """Create a latency results table."""
        if not latency_results:
            return pd.DataFrame()

        table_data = []
        for model_name, model_latency in latency_results.items():
            row = {"Model": model_name}

            if "ttc" in model_latency and isinstance(model_latency["ttc"], dict):
                ttc_data = model_latency["ttc"]

                for metric in ["p50", "p95"]:
                    value = ttc_data.get(metric, float("nan"))
                    row[f"TTC {metric.upper()} (ms)"] = "N/A" if pd.isna(value) else f"{value:.1f}"
            else:
                row["TTC P50 (ms)"] = "N/A"
                row["TTC P95 (ms)"] = "N/A"

            table_data.append(row)

        return pd.DataFrame(table_data)

    def _save_summary_tables(
        self, benchmark_dir: Path, metrics_by_model: dict[str, dict[str, float]], latency_results: dict[str, dict[str, Any]]
    ) -> None:
        """Save summary tables to a file."""
        output_file = benchmark_dir / "benchmark_summary_tables.txt"

        try:
            perf_table = self._create_performance_table(metrics_by_model)
            latency_table = self._create_latency_table(latency_results)

            with open(output_file, "w") as f:
                f.write("BENCHMARK SUMMARY TABLES\n")
                f.write("=" * 80 + "\n\n")

                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 80 + "\n")
                if not perf_table.empty:
                    f.write(perf_table.to_string(index=False))
                else:
                    f.write("No data available")
                f.write("\n\n")

                f.write("LATENCY RESULTS (Time to Completion)\n")
                f.write("-" * 80 + "\n")
                if not latency_table.empty:
                    f.write(latency_table.to_string(index=False))
                else:
                    f.write("No data available")
                f.write("\n\n")

            logger.info("Summary tables saved to: %s", output_file)

        except Exception as e:
            logger.error("Failed to save summary tables: %s", e)

    def _save_results_jsonl(self, results: list[SampleResult], filepath: Path) -> None:
        """Save results in JSONL format."""
        with filepath.open("w", encoding="utf-8") as f:
            for result in results:
                result_dict = {
                    "id": result.id,
                    "expected_triggers": result.expected_triggers,
                    "triggered": result.triggered,
                    "details": result.details or {},
                }
                f.write(json.dumps(result_dict) + "\n")

    def _save_metrics_json(self, metrics_by_model: dict[str, dict[str, float]], filepath: Path) -> None:
        """Save performance metrics in JSON format."""
        with filepath.open("w") as f:
            json.dump(metrics_by_model, f, indent=2)

    def _save_latency_json(self, latency_results: dict[str, dict[str, Any]], filepath: Path) -> None:
        """Save latency results in JSON format."""
        with filepath.open("w") as f:
            json.dump(latency_results, f, indent=2)

    def _save_benchmark_summary(
        self,
        filepath: Path,
        guardrail_name: str,
        results_by_model: dict[str, list[SampleResult]],
        metrics_by_model: dict[str, dict[str, float]],
        latency_results: dict[str, dict[str, Any]],
        dataset_size: int,
        latency_iterations: int,
    ) -> None:
        """Save human-readable benchmark summary."""
        with filepath.open("w", encoding="utf-8") as f:
            f.write("Guardrail Benchmark Results\n")
            f.write("===========================\n\n")
            f.write(f"Guardrail: {guardrail_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Dataset size: {dataset_size} samples\n")
            f.write(f"Latency iterations: {latency_iterations}\n\n")

            f.write(f"Models evaluated: {', '.join(results_by_model.keys())}\n\n")

            f.write("Performance Metrics Summary:\n")
            f.write("---------------------------\n")
            for model_name, metrics in metrics_by_model.items():
                f.write(f"\n{model_name}:\n")
                for metric_name, value in metrics.items():
                    if not isinstance(value, float) or not value != value:  # Check for NaN
                        f.write(f"  {metric_name}: {value}\n")

            f.write("\nLatency Summary:\n")
            f.write("----------------\n")
            for model_name, latency_data in latency_results.items():
                f.write(f"\n{model_name}:\n")
                if "error" in latency_data:
                    f.write(f"  Error: {latency_data['error']}\n")
                else:
                    ttft = latency_data["ttft"]
                    ttc = latency_data["ttc"]
                    f.write(f"  TTFT P50: {ttft['p50']:.1f}ms, P95: {ttft['p95']:.1f}ms\n")
                    f.write(f"  TTC P50: {ttc['p50']:.1f}ms, P95: {ttc['p95']:.1f}ms\n")
