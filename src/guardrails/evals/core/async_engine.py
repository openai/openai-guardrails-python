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


def _safe_getattr(obj: Any, key: str, default: Any = None) -> Any:
    """Get attribute or dict key defensively."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_conversation_payload(payload: Any) -> list[Any] | None:
    """Normalize decoded sample payload into a conversation list if possible."""
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for candidate_key in ("messages", "conversation", "conversation_history"):
            value = payload.get(candidate_key)
            if isinstance(value, list):
                return value

    return None


def _parse_conversation_payload(data: str) -> list[Any]:
    """Attempt to parse sample data into a conversation history list.

    If data is JSON, tries to extract conversation from it.
    If data is a plain string, wraps it as a single user message.
    Always returns a list (never None).
    """
    try:
        payload = json.loads(data)
        normalized = _normalize_conversation_payload(payload)
        if normalized:
            return normalized
        # JSON parsed but not a conversation format - treat as user message
        return [{"role": "user", "content": data}]
    except json.JSONDecodeError:
        # Not JSON - treat as a plain user message
        return [{"role": "user", "content": data}]


def _annotate_prompt_injection_result(result: Any, turn_index: int, message: Any) -> None:
    """Annotate guardrail result with incremental evaluation metadata."""
    role = _safe_getattr(message, "role")
    msg_type = _safe_getattr(message, "type")
    info = result.info
    info["last_checked_turn_index"] = turn_index
    if role is not None:
        info["last_checked_role"] = role
    if msg_type is not None:
        info["last_checked_type"] = msg_type
    if result.tripwire_triggered:
        info["trigger_turn_index"] = turn_index
        if role is not None:
            info["trigger_role"] = role
        if msg_type is not None:
            info["trigger_type"] = msg_type
        info["trigger_message"] = message


async def _run_incremental_prompt_injection(
    client: GuardrailsAsyncOpenAI,
    conversation_history: list[Any],
) -> list[Any]:
    """Run prompt injection guardrail incrementally over a conversation."""
    latest_results: list[Any] = []

    for turn_index in range(len(conversation_history)):
        current_history = conversation_history[: turn_index + 1]
        stage_results = await client._run_stage_guardrails(
            stage_name="output",
            text="",
            conversation_history=current_history,
            suppress_tripwire=True,
        )

        latest_results = stage_results or latest_results

        for result in stage_results:
            if result.info.get("guardrail_name") == "Prompt Injection Detection":
                _annotate_prompt_injection_result(result, turn_index, current_history[-1])
                if result.tripwire_triggered:
                    return stage_results

    return latest_results


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
            # Detect if this sample requires conversation history
            conversation_aware_names = {"Prompt Injection Detection", "Jailbreak"}
            needs_conversation_history = any(name in sample.expected_triggers for name in conversation_aware_names)

            if needs_conversation_history:
                try:
                    # Parse conversation history from sample.data
                    # Handles JSON conversations, plain strings (wraps as user message), etc.
                    conversation_history = _parse_conversation_payload(sample.data)

                    # Create a minimal guardrails config for conversation-aware checks
                    minimal_config = {
                        "version": 1,
                        "output": {
                            "guardrails": [
                                {
                                    "name": guardrail.definition.name,
                                    "config": (guardrail.config.__dict__ if hasattr(guardrail.config, "__dict__") else guardrail.config),
                                }
                                for guardrail in self.guardrails
                                if guardrail.definition.name in conversation_aware_names
                            ],
                        },
                    }

                    # Create a temporary GuardrailsAsyncOpenAI client for conversation-aware guardrails
                    temp_client = GuardrailsAsyncOpenAI(
                        config=minimal_config,
                        api_key=getattr(context.guardrail_llm, "api_key", None) or "fake-key-for-eval",
                    )

                    # Normalize conversation history using the client's normalization
                    normalized_conversation = temp_client._normalize_conversation(conversation_history)

                    # Prompt injection detection uses incremental evaluation (per turn)
                    # Jailbreak uses full conversation evaluation
                    if "Prompt Injection Detection" in sample.expected_triggers:
                        results = await _run_incremental_prompt_injection(
                            temp_client,
                            normalized_conversation,
                        )
                    else:
                        # Jailbreak or other conversation-aware guardrails
                        results = await temp_client._run_stage_guardrails(
                            stage_name="output",
                            text="",
                            conversation_history=normalized_conversation,
                            suppress_tripwire=True,
                        )
                except (json.JSONDecodeError, TypeError, ValueError) as e:
                    logger.error(
                        "Failed to parse conversation history for conversation-aware guardrail sample %s: %s",
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
                        "Failed to create conversation context for guardrail sample %s: %s",
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
                # Standard sample (no conversation history needed)
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
