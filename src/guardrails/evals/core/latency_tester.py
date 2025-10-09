"""Latency testing for guardrail benchmarking.

This module implements end-to-end guardrail latency testing for different models.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from guardrails.runtime import instantiate_guardrails

from .async_engine import AsyncRunEngine
from .types import Context, Sample

logger = logging.getLogger(__name__)


class LatencyTester:
    """Tests end-to-end guardrail latency for different models."""

    def __init__(self, iterations: int = 20) -> None:
        """Initialize the latency tester.

        Args:
            iterations: Number of samples to time per model
        """
        self.iterations = iterations

    def calculate_latency_stats(self, times: list[float]) -> dict[str, float]:
        """Calculate latency statistics from a list of times.

        Args:
            times: List of latency times in seconds

        Returns:
            Dictionary with P50, P95, mean, and std dev (in milliseconds)
        """
        if not times:
            return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan"), "std": float("nan")}

        times_ms = np.array(times) * 1000  # Convert to milliseconds

        return {
            "p50": float(np.percentile(times_ms, 50)),
            "p95": float(np.percentile(times_ms, 95)),
            "mean": float(np.mean(times_ms)),
            "std": float(np.std(times_ms)),
        }

    async def test_guardrail_latency_for_model(
        self,
        context: Context,
        stage_bundle: Any,
        samples: list[Sample],
        iterations: int,
        *,
        desc: str | None = None,
    ) -> dict[str, Any]:
        """Measure end-to-end guardrail latency per sample for a single model.

        Args:
            context: Evaluation context with LLM client
            stage_bundle: Stage bundle configured for the specific model
            samples: Full dataset samples
            iterations: Number of samples to time (uses first N samples)
            desc: Optional tqdm description

        Returns:
            Dictionary with latency statistics and raw times
        """
        guardrails = instantiate_guardrails(stage_bundle)
        engine = AsyncRunEngine(guardrails)

        num = min(iterations, len(samples))
        if num <= 0:
            return self._empty_latency_result()

        ttc_times: list[float] = []
        bar_desc = desc or "Latency"

        with tqdm(total=num, desc=bar_desc, leave=True) as pbar:
            for i in range(num):
                sample = samples[i]
                start = time.perf_counter()
                await engine.run(context, [sample], batch_size=1, desc=None)
                ttc = time.perf_counter() - start
                ttc_times.append(ttc)
                pbar.update(1)

        ttc_stats = self.calculate_latency_stats(ttc_times)

        return {
            "ttft": ttc_stats,  # TTFT same as TTC at guardrail level
            "ttc": ttc_stats,
            "raw_times": {"ttft": ttc_times, "ttc": ttc_times},
            "iterations": len(ttc_times),
        }

    def _empty_latency_result(self) -> dict[str, Any]:
        """Return empty latency result structure."""
        empty_stats = {"p50": float("nan"), "p95": float("nan"), "mean": float("nan"), "std": float("nan")}
        return {
            "ttft": empty_stats,
            "ttc": empty_stats,
            "raw_times": {"ttft": [], "ttc": []},
            "iterations": 0,
        }
