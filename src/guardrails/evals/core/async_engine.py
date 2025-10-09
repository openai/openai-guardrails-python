"""Async run engine for guardrail evaluation.

This module provides an asynchronous engine for running guardrail checks on evaluation samples.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from tqdm import tqdm

from guardrails import GuardrailsAsyncOpenAI, run_guardrails

from .types import Context, RunEngine, Sample, SampleResult

logger = logging.getLogger(__name__)


class AsyncRunEngine(RunEngine):
    """Runs guardrail evaluations asynchronously."""

    def __init__(self, guardrails: list[Any]) -> None:
        """Initialize the run engine.

        Args:
            guardrails: List of configured guardrails to evaluate
        """
        self.guardrails = guardrails
        self.guardrail_names = [g.definition.name for g in guardrails]
        logger.info(
            "Initialized engine with %d guardrails: %s",
            len(self.guardrail_names),
            ", ".join(self.guardrail_names),
        )

    async def run(
        self,
        context: Context,
        samples: list[Sample],
        batch_size: int,
        desc: str | None = None,
    ) -> list[SampleResult]:
        """Run evaluations on samples in batches.

        Args:
            context: Evaluation context with LLM client
            samples: List of samples to evaluate
            batch_size: Number of samples to process in parallel
            desc: Description for the tqdm progress bar

        Returns:
            List of evaluation results

        Raises:
            ValueError: If batch_size is less than 1
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if not samples:
            logger.warning("No samples provided for evaluation")
            return []

        logger.info(
            "Starting evaluation of %d samples with batch size %d",
            len(samples),
            batch_size,
        )

        results: list[SampleResult] = []
        use_progress = bool(desc) and len(samples) > 1

        if use_progress:
            with tqdm(total=len(samples), desc=desc, leave=True) as progress:
                results = await self._run_with_progress(context, samples, batch_size, progress)
        else:
            results = await self._run_without_progress(context, samples, batch_size)

        logger.info("Evaluation completed. Processed %d samples", len(results))
        return results

    async def _run_with_progress(self, context: Context, samples: list[Sample], batch_size: int, progress: tqdm) -> list[SampleResult]:
        """Run evaluation with progress bar."""
        results = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            batch_results = await self._process_batch(context, batch)
            results.extend(batch_results)
            progress.update(len(batch))
        return results

    async def _run_without_progress(self, context: Context, samples: list[Sample], batch_size: int) -> list[SampleResult]:
        """Run evaluation without progress bar."""
        results = []
        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            batch_results = await self._process_batch(context, batch)
            results.extend(batch_results)
        return results

    async def _process_batch(self, context: Context, batch: list[Sample]) -> list[SampleResult]:
        """Process a batch of samples."""
        batch_results = await asyncio.gather(
            *(self._evaluate_sample(context, sample) for sample in batch),
            return_exceptions=True,
        )

        # Handle any exceptions from the batch
        results = []
        for sample, result in zip(batch, batch_results, strict=False):
            if isinstance(result, Exception):
                logger.error("Sample %s failed: %s", sample.id, str(result))
                result = SampleResult(
                    id=sample.id,
                    expected_triggers=sample.expected_triggers,
                    triggered=dict.fromkeys(self.guardrail_names, False),
                    details={"error": str(result)},
                )
            results.append(result)

        return results

    async def _evaluate_sample(self, context: Context, sample: Sample) -> SampleResult:
        """Evaluate a single sample against all guardrails.

        Args:
            context: Evaluation context with LLM client
            sample: Sample to evaluate

        Returns:
            Evaluation result for the sample
        """
        try:
            # Detect if this is a prompt injection detection sample and use GuardrailsAsyncOpenAI
            if "Prompt Injection Detection" in sample.expected_triggers:
                try:
                    # Parse conversation history from sample.data (JSON string)
                    conversation_history = json.loads(sample.data)
                    logger.debug(
                        "Parsed conversation history for prompt injection detection sample %s: %d items",
                        sample.id,
                        len(conversation_history),
                    )

                    # Use GuardrailsAsyncOpenAI with a minimal config to get proper context
                    # Create a minimal guardrails config for the prompt injection detection check
                    minimal_config = {
                        "version": 1,
                        "output": {
                            "version": 1,
                            "guardrails": [
                                {
                                    "name": guardrail.definition.name,
                                    "config": (guardrail.config.__dict__ if hasattr(guardrail.config, "__dict__") else guardrail.config),
                                }
                                for guardrail in self.guardrails
                                if guardrail.definition.name == "Prompt Injection Detection"
                            ],
                        },
                    }

                    # Create a temporary GuardrailsAsyncOpenAI client to run the prompt injection detection check
                    temp_client = GuardrailsAsyncOpenAI(
                        config=minimal_config,
                        api_key=getattr(context.guardrail_llm, "api_key", None) or "fake-key-for-eval",
                    )

                    # Use the client's _run_stage_guardrails method with conversation history
                    results = await temp_client._run_stage_guardrails(
                        stage_name="output",
                        text="",  # Prompt injection detection doesn't use text data
                        conversation_history=conversation_history,
                        suppress_tripwire=True,
                    )
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(
                        "Failed to parse conversation history for prompt injection detection sample %s: %s",
                        sample.id,
                        e,
                    )
                    # Fall back to standard evaluation
                    results = await run_guardrails(
                        ctx=context,
                        data=sample.data,
                        media_type="text/plain",
                        guardrails=self.guardrails,
                        suppress_tripwire=True,  # Collect all results, don't stop on tripwire
                    )
                except Exception as e:
                    logger.error(
                        "Failed to create prompt injection detection context for sample %s: %s",
                        sample.id,
                        e,
                    )
                    # Fall back to standard evaluation
                    results = await run_guardrails(
                        ctx=context,
                        data=sample.data,
                        media_type="text/plain",
                        guardrails=self.guardrails,
                        suppress_tripwire=True,  # Collect all results, don't stop on tripwire
                    )
            else:
                # Standard non-prompt injection detection sample
                results = await run_guardrails(
                    ctx=context,
                    data=sample.data,
                    media_type="text/plain",
                    guardrails=self.guardrails,
                    suppress_tripwire=True,  # Collect all results, don't stop on tripwire
                )

            triggered: dict[str, bool] = dict.fromkeys(self.guardrail_names, False)
            details: dict[str, Any] = {}

            for result in results:
                guardrail_name = result.info.get("guardrail_name", "unknown")
                if guardrail_name in self.guardrail_names:
                    triggered[guardrail_name] = result.tripwire_triggered
                    details[guardrail_name] = result.info
                else:
                    logger.warning("Unknown guardrail name: %s", guardrail_name)

            return SampleResult(
                id=sample.id,
                expected_triggers=sample.expected_triggers,
                triggered=triggered,
                details=details,
            )

        except Exception as e:
            logger.error("Error evaluating sample %s: %s", sample.id, str(e))
            return SampleResult(
                id=sample.id,
                expected_triggers=sample.expected_triggers,
                triggered=dict.fromkeys(self.guardrail_names, False),
                details={"error": str(e)},
            )
