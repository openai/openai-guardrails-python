"""Guardrail evaluation runner and CLI.

This script provides a command-line interface and class for running guardrail evaluations on datasets.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

try:
    from openai import AsyncAzureOpenAI
except ImportError:
    AsyncAzureOpenAI = None  # type: ignore


from guardrails import instantiate_guardrails, load_pipeline_bundles
from guardrails._openai_utils import prepare_openai_kwargs
from guardrails.evals.core import (
    AsyncRunEngine,
    BenchmarkMetricsCalculator,
    BenchmarkReporter,
    BenchmarkVisualizer,
    GuardrailMetricsCalculator,
    JsonlDatasetLoader,
    JsonResultsReporter,
    LatencyTester,
)
from guardrails.evals.core.types import Context

logger = logging.getLogger(__name__)

# Default models for benchmark mode
DEFAULT_BENCHMARK_MODELS = [
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
]
DEFAULT_BATCH_SIZE = 32
DEFAULT_LATENCY_ITERATIONS = 50
VALID_STAGES = {"pre_flight", "input", "output"}


class GuardrailEval:
    """Class for running guardrail evaluations."""

    def __init__(
        self,
        config_path: Path,
        dataset_path: Path,
        stages: Sequence[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        output_dir: Path = Path("results"),
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        azure_api_version: str | None = None,
        mode: str = "evaluate",
        models: Sequence[str] | None = None,
        latency_iterations: int = DEFAULT_LATENCY_ITERATIONS,
    ) -> None:
        """Initialize the evaluator.

        Args:
            config_path: Path to pipeline configuration file.
            dataset_path: Path to evaluation dataset (JSONL).
            stages: Specific stages to evaluate (pre_flight, input, output).
            batch_size: Number of samples to process in parallel.
            output_dir: Directory to save evaluation results.
            api_key: API key for OpenAI, Azure OpenAI, or OpenAI-compatible API.
            base_url: Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1).
            azure_endpoint: Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com).
            azure_api_version: Azure OpenAI API version (e.g., 2025-01-01-preview).
            mode: Evaluation mode ("evaluate" or "benchmark").
            models: Models to test in benchmark mode.
            latency_iterations: Number of iterations for latency testing.
        """
        self._validate_inputs(config_path, dataset_path, batch_size, mode, latency_iterations)

        self.config_path = config_path
        self.dataset_path = dataset_path
        self.stages = stages
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.api_key = api_key
        self.base_url = base_url
        self.azure_endpoint = azure_endpoint
        self.azure_api_version = azure_api_version or "2025-01-01-preview"
        self.mode = mode
        self.models = models or DEFAULT_BENCHMARK_MODELS
        self.latency_iterations = latency_iterations

        # Validate Azure configuration
        if azure_endpoint and not AsyncAzureOpenAI:
            raise ValueError("Azure OpenAI support requires openai>=1.0.0. Please upgrade: pip install --upgrade openai")

    def _validate_inputs(self, config_path: Path, dataset_path: Path, batch_size: int, mode: str, latency_iterations: int) -> None:
        """Validate input parameters."""
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        if not dataset_path.exists():
            raise ValueError(f"Dataset file not found: {dataset_path}")

        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got: {batch_size}")

        if mode not in ("evaluate", "benchmark"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'evaluate' or 'benchmark'")

        if latency_iterations <= 0:
            raise ValueError(f"Latency iterations must be positive, got: {latency_iterations}")

    async def run(self) -> None:
        """Run the evaluation pipeline for all specified stages."""
        try:
            if self.mode == "benchmark":
                await self._run_benchmark()
            else:
                await self._run_evaluation()
        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            raise

    async def _run_evaluation(self) -> None:
        """Run standard evaluation mode."""
        pipeline_bundles = load_pipeline_bundles(self.config_path)
        stages_to_evaluate = self._get_valid_stages(pipeline_bundles)

        if not stages_to_evaluate:
            raise ValueError("No valid stages found in configuration")

        logger.info("Evaluating stages: %s", ", ".join(stages_to_evaluate))

        loader = JsonlDatasetLoader()
        samples = loader.load(self.dataset_path)
        logger.info("Loaded %d samples from dataset", len(samples))

        context = self._create_context()
        calculator = GuardrailMetricsCalculator()
        reporter = JsonResultsReporter()

        all_results = {}
        all_metrics = {}

        for stage in stages_to_evaluate:
            logger.info("Starting %s stage evaluation", stage)

            try:
                stage_results = await self._evaluate_single_stage(stage, pipeline_bundles, samples, context, calculator)

                if stage_results:
                    all_results[stage] = stage_results["results"]
                    all_metrics[stage] = stage_results["metrics"]
                    logger.info("Completed %s stage evaluation", stage)
                else:
                    logger.warning("Stage '%s' evaluation returned no results", stage)

            except Exception as e:
                logger.error("Failed to evaluate stage '%s': %s", stage, e)

        if not all_results:
            raise ValueError("No stages were successfully evaluated")

        reporter.save_multi_stage(all_results, all_metrics, self.output_dir)
        logger.info("Evaluation completed. Results saved to: %s", self.output_dir)

    async def _run_benchmark(self) -> None:
        """Run benchmark mode comparing multiple models."""
        logger.info("Running benchmark mode with models: %s", ", ".join(self.models))

        pipeline_bundles = load_pipeline_bundles(self.config_path)
        stage_to_test, guardrail_name = self._get_benchmark_target(pipeline_bundles)

        # Validate guardrail has model configuration
        stage_bundle = getattr(pipeline_bundles, stage_to_test)
        if not self._has_model_configuration(stage_bundle):
            raise ValueError(
                f"Guardrail '{guardrail_name}' does not have a model configuration. "
                "Benchmark mode requires LLM-based guardrails with configurable models."
            )

        logger.info("Benchmarking guardrail '%s' from stage '%s'", guardrail_name, stage_to_test)

        loader = JsonlDatasetLoader()
        samples = loader.load(self.dataset_path)
        logger.info("Loaded %d samples for benchmarking", len(samples))

        context = self._create_context()
        benchmark_calculator = BenchmarkMetricsCalculator()
        basic_calculator = GuardrailMetricsCalculator()
        benchmark_reporter = BenchmarkReporter(self.output_dir)

        # Run benchmark for all models
        results_by_model, metrics_by_model = await self._benchmark_all_models(
            stage_to_test, guardrail_name, samples, context, benchmark_calculator, basic_calculator
        )

        # Run latency testing
        logger.info("Running latency tests for all models")
        latency_results = await self._run_latency_tests(stage_to_test, samples)

        # Save benchmark results
        benchmark_dir = benchmark_reporter.save_benchmark_results(
            results_by_model, metrics_by_model, latency_results, guardrail_name, len(samples), self.latency_iterations
        )

        # Create visualizations
        logger.info("Generating visualizations")
        visualizer = BenchmarkVisualizer(benchmark_dir / "graphs")
        visualization_files = visualizer.create_all_visualizations(
            results_by_model, metrics_by_model, latency_results, guardrail_name, samples[0].expected_triggers if samples else {}
        )

        logger.info("Benchmark completed. Results saved to: %s", benchmark_dir)
        logger.info("Generated %d visualizations", len(visualization_files))

    def _has_model_configuration(self, stage_bundle) -> bool:
        """Check if the guardrail has a model configuration."""
        if not stage_bundle.guardrails:
            return False

        guardrail_config = stage_bundle.guardrails[0].config
        if not guardrail_config:
            return False

        if isinstance(guardrail_config, dict) and "model" in guardrail_config:
            return True
        elif hasattr(guardrail_config, "model"):
            return True

        return False

    async def _run_latency_tests(self, stage_to_test: str, samples: list) -> dict[str, Any]:
        """Run latency tests for all models."""
        latency_results = {}
        latency_tester = LatencyTester(iterations=self.latency_iterations)

        for model in self.models:
            model_stage_bundle = self._create_model_specific_stage_bundle(getattr(load_pipeline_bundles(self.config_path), stage_to_test), model)
            model_context = self._create_context()
            latency_results[model] = await latency_tester.test_guardrail_latency_for_model(
                model_context,
                model_stage_bundle,
                samples,
                self.latency_iterations,
                desc=f"Testing latency: {model}",
            )

        return latency_results

    def _create_context(self) -> Context:
        """Create evaluation context with OpenAI client.

        Supports OpenAI, Azure OpenAI, and OpenAI-compatible APIs.
        Used for both evaluation and benchmark modes.

        Returns:
            Context with configured AsyncOpenAI or AsyncAzureOpenAI client.
        """
        # Azure OpenAI
        if self.azure_endpoint:
            if not AsyncAzureOpenAI:
                raise ValueError("Azure OpenAI support requires openai>=1.0.0. Please upgrade: pip install --upgrade openai")

            azure_kwargs = {
                "azure_endpoint": self.azure_endpoint,
                "api_version": self.azure_api_version,
            }
            if self.api_key:
                azure_kwargs["api_key"] = self.api_key

            guardrail_llm = AsyncAzureOpenAI(**prepare_openai_kwargs(azure_kwargs))
            logger.info("Created Azure OpenAI client for endpoint: %s", self.azure_endpoint)
        # OpenAI or OpenAI-compatible API
        else:
            openai_kwargs = {}
            if self.api_key:
                openai_kwargs["api_key"] = self.api_key
            if self.base_url:
                openai_kwargs["base_url"] = self.base_url
                logger.info("Created OpenAI-compatible client for base_url: %s", self.base_url)

            guardrail_llm = AsyncOpenAI(**prepare_openai_kwargs(openai_kwargs))

        return Context(guardrail_llm=guardrail_llm)

    def _is_valid_stage(self, pipeline_bundles, stage: str) -> bool:
        """Check if a stage has valid guardrails configured.

        Args:
            pipeline_bundles: Pipeline bundles object.
            stage: Stage name to check.

        Returns:
            True if stage exists and has guardrails configured.
        """
        if not hasattr(pipeline_bundles, stage):
            return False

        stage_bundle = getattr(pipeline_bundles, stage)
        return stage_bundle is not None and hasattr(stage_bundle, "guardrails") and bool(stage_bundle.guardrails)

    def _create_model_specific_stage_bundle(self, stage_bundle, model: str):
        """Create a deep copy of the stage bundle with model-specific configuration."""
        try:
            modified_bundle = copy.deepcopy(stage_bundle)
        except Exception as e:
            logger.error("Failed to create deep copy of stage bundle: %s", e)
            raise ValueError(f"Failed to create deep copy of stage bundle: {e}") from e

        logger.info("Creating model-specific configuration for model: %s", model)

        guardrails_updated = 0
        for guardrail in modified_bundle.guardrails:
            try:
                if hasattr(guardrail, "config") and guardrail.config:
                    if isinstance(guardrail.config, dict) and "model" in guardrail.config:
                        original_model = guardrail.config["model"]
                        guardrail.config["model"] = model
                        logger.info("Updated guardrail '%s' model from '%s' to '%s'", guardrail.name, original_model, model)
                        guardrails_updated += 1
                    elif hasattr(guardrail.config, "model"):
                        original_model = getattr(guardrail.config, "model", "unknown")
                        guardrail.config.model = model
                        logger.info("Updated guardrail '%s' model from '%s' to '%s'", guardrail.name, original_model, model)
                        guardrails_updated += 1
            except Exception as e:
                logger.error("Failed to update guardrail '%s' configuration: %s", guardrail.name, e)
                raise ValueError(f"Failed to update guardrail '{guardrail.name}' configuration: {e}") from e

        if guardrails_updated == 0:
            logger.warning("No guardrails with model configuration were found")
        else:
            logger.info("Successfully updated %d guardrail(s) for model: %s", guardrails_updated, model)

        return modified_bundle

    def _get_valid_stages(self, pipeline_bundles) -> list[str]:
        """Get list of valid stages to evaluate."""
        if self.stages is None:
            # Auto-detect all valid stages
            available_stages = [stage for stage in VALID_STAGES if self._is_valid_stage(pipeline_bundles, stage)]

            if not available_stages:
                raise ValueError("No valid stages found in configuration")

            logger.info("No stages specified, evaluating all available stages: %s", ", ".join(available_stages))
            return available_stages
        else:
            # Validate requested stages
            valid_requested_stages = []
            for stage in self.stages:
                if stage not in VALID_STAGES:
                    logger.warning("Invalid stage '%s', skipping", stage)
                    continue

                if not self._is_valid_stage(pipeline_bundles, stage):
                    logger.warning("Stage '%s' not found or has no guardrails configured, skipping", stage)
                    continue

                valid_requested_stages.append(stage)

            if not valid_requested_stages:
                raise ValueError("No valid stages found in configuration")

            return valid_requested_stages

    async def _evaluate_single_stage(
        self, stage: str, pipeline_bundles, samples: list, context: Context, calculator: GuardrailMetricsCalculator
    ) -> dict[str, Any] | None:
        """Evaluate a single pipeline stage."""
        try:
            stage_bundle = getattr(pipeline_bundles, stage)
            guardrails = instantiate_guardrails(stage_bundle)

            engine = AsyncRunEngine(guardrails)

            stage_results = await engine.run(context, samples, self.batch_size, desc=f"Evaluating {stage} stage")

            stage_metrics = calculator.calculate(stage_results)

            return {"results": stage_results, "metrics": stage_metrics}

        except Exception as e:
            logger.error("Failed to evaluate stage '%s': %s", stage, e)
            return None

    def _get_benchmark_target(self, pipeline_bundles) -> tuple[str, str]:
        """Get the stage and guardrail to benchmark."""
        if self.stages:
            stage_to_test = self.stages[0]
            if not self._is_valid_stage(pipeline_bundles, stage_to_test):
                raise ValueError(f"Stage '{stage_to_test}' has no guardrails configured")
        else:
            # Find first valid stage
            stage_to_test = next((stage for stage in VALID_STAGES if self._is_valid_stage(pipeline_bundles, stage)), None)
            if not stage_to_test:
                raise ValueError("No valid stage found for benchmarking")

        stage_bundle = getattr(pipeline_bundles, stage_to_test)
        guardrail_name = stage_bundle.guardrails[0].name

        return stage_to_test, guardrail_name

    async def _benchmark_all_models(
        self,
        stage_to_test: str,
        guardrail_name: str,
        samples: list,
        context: Context,
        benchmark_calculator: BenchmarkMetricsCalculator,
        basic_calculator: GuardrailMetricsCalculator,
    ) -> tuple[dict[str, list], dict[str, dict]]:
        """Benchmark all models for the specified stage and guardrail."""
        pipeline_bundles = load_pipeline_bundles(self.config_path)
        stage_bundle = getattr(pipeline_bundles, stage_to_test)

        results_by_model = {}
        metrics_by_model = {}

        for i, model in enumerate(self.models, 1):
            logger.info("Testing model %d/%d: %s", i, len(self.models), model)

            try:
                modified_stage_bundle = self._create_model_specific_stage_bundle(stage_bundle, model)

                model_results = await self._benchmark_single_model(
                    model, modified_stage_bundle, samples, context, guardrail_name, benchmark_calculator, basic_calculator
                )

                if model_results:
                    results_by_model[model] = model_results["results"]
                    metrics_by_model[model] = model_results["metrics"]
                    logger.info("Completed benchmarking for model %s (%d/%d)", model, i, len(self.models))
                else:
                    logger.warning("Model %s benchmark returned no results (%d/%d)", model, i, len(self.models))
                    results_by_model[model] = []
                    metrics_by_model[model] = {}

            except Exception as e:
                logger.error("Failed to benchmark model %s (%d/%d): %s", model, i, len(self.models), e)
                results_by_model[model] = []
                metrics_by_model[model] = {}

        # Log summary
        successful_models = [model for model, results in results_by_model.items() if results]
        failed_models = [model for model, results in results_by_model.items() if not results]

        logger.info("BENCHMARK SUMMARY")
        logger.info("Successful models: %s", ", ".join(successful_models) if successful_models else "None")
        if failed_models:
            logger.warning("Failed models: %s", ", ".join(failed_models))
        logger.info("Total models tested: %d", len(self.models))

        return results_by_model, metrics_by_model

    async def _benchmark_single_model(
        self,
        model: str,
        stage_bundle,
        samples: list,
        context: Context,
        guardrail_name: str,
        benchmark_calculator: BenchmarkMetricsCalculator,
        basic_calculator: GuardrailMetricsCalculator,
    ) -> dict[str, Any] | None:
        """Benchmark a single model."""
        try:
            model_context = self._create_context()

            guardrails = instantiate_guardrails(stage_bundle)
            engine = AsyncRunEngine(guardrails)
            model_results = await engine.run(model_context, samples, self.batch_size, desc=f"Benchmarking {model}")

            guardrail_config = stage_bundle.guardrails[0].config if stage_bundle.guardrails else None

            advanced_metrics = benchmark_calculator.calculate_advanced_metrics(model_results, guardrail_name, guardrail_config)

            basic_metrics = basic_calculator.calculate(model_results)

            if guardrail_name in basic_metrics:
                guardrail_metrics = basic_metrics[guardrail_name]
                basic_metrics_dict = {
                    "precision": guardrail_metrics.precision,
                    "recall": guardrail_metrics.recall,
                    "f1_score": guardrail_metrics.f1_score,
                    "true_positives": guardrail_metrics.true_positives,
                    "false_positives": guardrail_metrics.false_positives,
                    "false_negatives": guardrail_metrics.false_negatives,
                    "true_negatives": guardrail_metrics.true_negatives,
                    "total_samples": guardrail_metrics.total_samples,
                }
            else:
                basic_metrics_dict = {}

            combined_metrics = {**basic_metrics_dict, **advanced_metrics}

            return {"results": model_results, "metrics": combined_metrics}

        except Exception as e:
            logger.error("Failed to benchmark model %s: %s", model, e)
            return None


def main() -> None:
    """Run the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Run guardrail evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard evaluation of all stages
  guardrails-evals --config-path config.json --dataset-path data.jsonl

  # Multi-stage evaluation
  guardrails-evals --config-path config.json --dataset-path data.jsonl --stages pre_flight input

  # Benchmark mode with OpenAI models
  guardrails-evals --config-path config.json --dataset-path data.jsonl --mode benchmark --models gpt-5 gpt-5-mini

  # Azure OpenAI benchmark
  guardrails-evals --config-path config.json --dataset-path data.jsonl --mode benchmark \\
    --azure-endpoint https://your-resource.openai.azure.com --api-key your-key \\
    --models gpt-4o gpt-4o-mini

  # Ollama local models
  guardrails-evals --config-path config.json --dataset-path data.jsonl --mode benchmark \\
    --base-url http://localhost:11434/v1 --api-key fake-key --models llama3 mistral

  # vLLM or other OpenAI-compatible API
  guardrails-evals --config-path config.json --dataset-path data.jsonl --mode benchmark \\
    --base-url http://your-server:8000/v1 --api-key your-key --models your-model

  # Module execution during local development
  python -m guardrails.evals.guardrail_evals --config-path config.json --dataset-path data.jsonl
        """,
    )

    # Required arguments
    parser.add_argument(
        "--config-path",
        type=Path,
        required=True,
        help="Path to the guardrail config file",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Path to the evaluation dataset",
    )

    # Evaluation mode
    parser.add_argument(
        "--mode",
        choices=["evaluate", "benchmark"],
        default="evaluate",
        help="Evaluation mode: 'evaluate' for standard evaluation, 'benchmark' for model comparison (default: evaluate)",
    )

    # Optional evaluation arguments
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=VALID_STAGES,
        help="Pipeline stages to evaluate. If not specified, evaluates all stages found in config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of samples to process in parallel (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to save evaluation results (default: results)",
    )

    # API configuration
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for OpenAI, Azure OpenAI, or OpenAI-compatible API",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        help="Base URL for OpenAI-compatible API (e.g., http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--azure-endpoint",
        type=str,
        help="Azure OpenAI endpoint (e.g., https://your-resource.openai.azure.com)",
    )
    parser.add_argument(
        "--azure-api-version",
        type=str,
        default="2025-01-01-preview",
        help="Azure OpenAI API version (default: 2025-01-01-preview)",
    )

    # Benchmark-only arguments
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to test in benchmark mode (default: gpt-5, gpt-5-mini, gpt-5-nano, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano)",
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=DEFAULT_LATENCY_ITERATIONS,
        help=f"Number of iterations for latency testing in benchmark mode (default: {DEFAULT_LATENCY_ITERATIONS})",
    )

    args = parser.parse_args()

    # Validate arguments
    try:
        if not args.config_path.exists():
            print(f"❌ Error: Config file not found: {args.config_path}")
            sys.exit(1)

        if not args.dataset_path.exists():
            print(f"❌ Error: Dataset file not found: {args.dataset_path}")
            sys.exit(1)

        if args.batch_size <= 0:
            print(f"❌ Error: Batch size must be positive, got: {args.batch_size}")
            sys.exit(1)

        if args.latency_iterations <= 0:
            print(f"❌ Error: Latency iterations must be positive, got: {args.latency_iterations}")
            sys.exit(1)

        if args.stages:
            invalid_stages = [stage for stage in args.stages if stage not in VALID_STAGES]
            if invalid_stages:
                print(f"❌ Error: Invalid stages: {invalid_stages}. Valid stages are: {', '.join(VALID_STAGES)}")
                sys.exit(1)

        if args.mode == "benchmark" and args.stages and len(args.stages) > 1:
            print("⚠️  Warning: Benchmark mode only uses the first specified stage. Additional stages will be ignored.")

        # Validate provider configuration
        azure_endpoint = getattr(args, "azure_endpoint", None)
        base_url = getattr(args, "base_url", None)

        if azure_endpoint and base_url:
            print("❌ Error: Cannot specify both --azure-endpoint and --base-url. Choose one provider.")
            sys.exit(1)

        if azure_endpoint and not args.api_key:
            print("❌ Error: --api-key is required when using --azure-endpoint")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Error validating arguments: {e}")
        sys.exit(1)

    # Run evaluation
    try:
        print(f"🚀 Starting guardrail evaluation in {args.mode} mode...")
        print(f"   Config: {args.config_path}")
        print(f"   Dataset: {args.dataset_path}")
        print(f"   Output: {args.output_dir}")

        # Show provider configuration
        if getattr(args, "azure_endpoint", None):
            print(f"   Provider: Azure OpenAI ({args.azure_endpoint})")
        elif getattr(args, "base_url", None):
            print(f"   Provider: OpenAI-compatible API ({args.base_url})")
        else:
            print("   Provider: OpenAI")

        if args.mode == "benchmark":
            print(f"   Models: {', '.join(args.models or DEFAULT_BENCHMARK_MODELS)}")
            print(f"   Latency iterations: {args.latency_iterations}")

        eval = GuardrailEval(
            config_path=args.config_path,
            dataset_path=args.dataset_path,
            stages=args.stages,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            api_key=args.api_key,
            base_url=getattr(args, "base_url", None),
            azure_endpoint=getattr(args, "azure_endpoint", None),
            azure_api_version=getattr(args, "azure_api_version", None),
            mode=args.mode,
            models=args.models,
            latency_iterations=args.latency_iterations,
        )

        asyncio.run(eval.run())
        print("✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
