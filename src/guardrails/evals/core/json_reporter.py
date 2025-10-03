"""JSON results reporter for guardrail evaluation.

This module implements a reporter that saves evaluation results and metrics in JSON and JSONL formats.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TextIO

from .types import GuardrailMetrics, ResultsReporter, SampleResult

logger = logging.getLogger(__name__)


class JsonResultsReporter(ResultsReporter):
    """Reports evaluation results in JSON format."""

    def save(
        self,
        results: list[SampleResult],
        metrics: dict[str, GuardrailMetrics],
        output_dir: Path,
    ) -> None:
        """Save evaluation results to files.

        Args:
            results: List of evaluation results
            metrics: Dictionary of guardrail metrics
            output_dir: Directory to save results

        Raises:
            OSError: If there are any file I/O errors
            ValueError: If results or metrics are empty
        """
        if not results:
            raise ValueError("Cannot save empty results list")
        if not metrics:
            raise ValueError("Cannot save empty metrics dictionary")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / f"eval_run_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save per-sample results
            results_file = run_dir / "eval_results.jsonl"
            with results_file.open("w", encoding="utf-8") as f:
                self._write_results(f, results)

            # Save metrics
            metrics_file = run_dir / "eval_metrics.json"
            with metrics_file.open("w") as f:
                metrics_dict = {k: v.model_dump() for k, v in metrics.items()}
                json.dump(metrics_dict, f, indent=2)

            logger.info("Results saved to %s", results_file)
            logger.info("Metrics saved to %s", metrics_file)
            logger.info("Evaluation run saved to: %s", run_dir)

        except OSError as e:
            logger.error("Failed to save results: %s", str(e))
            raise

    def save_multi_stage(
        self,
        all_results: dict[str, list[SampleResult]],
        all_metrics: dict[str, dict[str, GuardrailMetrics]],
        output_dir: Path,
    ) -> None:
        """Save multi-stage evaluation results to files.

        Args:
            all_results: Dictionary mapping stage names to lists of results
            all_metrics: Dictionary mapping stage names to metrics dictionaries
            output_dir: Directory to save results

        Raises:
            OSError: If there are any file I/O errors
            ValueError: If results or metrics are empty
        """
        if not all_results:
            raise ValueError("Cannot save empty results dictionary")
        if not all_metrics:
            raise ValueError("Cannot save empty metrics dictionary")

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = output_dir / f"eval_run_{timestamp}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save per-stage results
            for stage, results in all_results.items():
                stage_results_file = run_dir / f"eval_results_{stage}.jsonl"
                with stage_results_file.open("w", encoding="utf-8") as f:
                    self._write_results(f, results)
                logger.info("Stage %s results saved to %s", stage, stage_results_file)

            # Save combined metrics
            metrics_file = run_dir / "eval_metrics.json"
            with metrics_file.open("w") as f:
                combined_metrics = {}
                for stage, metrics in all_metrics.items():
                    stage_metrics_dict = {k: v.model_dump() for k, v in metrics.items()}
                    combined_metrics[stage] = stage_metrics_dict

                json.dump(combined_metrics, f, indent=2)

            # Save run summary
            self._save_run_summary(run_dir, all_results)

            logger.info("Multi-stage metrics saved to %s", metrics_file)
            logger.info("Evaluation run saved to: %s", run_dir)

        except OSError as e:
            logger.error("Failed to save multi-stage results: %s", str(e))
            raise

    def _save_run_summary(self, run_dir: Path, all_results: dict[str, list[SampleResult]]) -> None:
        """Save run summary to file."""
        summary_file = run_dir / "run_summary.txt"
        with summary_file.open("w") as f:
            f.write("Guardrails Evaluation Run\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Stages evaluated: {', '.join(all_results.keys())}\n")
            f.write(f"Total samples: {len(next(iter(all_results.values())))}\n")
            f.write("\nStage breakdown:\n")
            for stage, results in all_results.items():
                f.write(f"  {stage}: {len(results)} samples\n")
            f.write("\nFiles created:\n")
            for stage in all_results.keys():
                f.write(f"  eval_results_{stage}.jsonl: Per-sample results for {stage} stage\n")
            f.write("  eval_metrics.json: Combined metrics for all stages\n")
            f.write("  run_summary.txt: This summary file\n")

        logger.info("Run summary saved to %s", summary_file)

    def _write_results(self, file: TextIO, results: list[SampleResult]) -> None:
        """Write results to file in JSONL format."""
        for result in results:
            file.write(result.model_dump_json() + "\n")
