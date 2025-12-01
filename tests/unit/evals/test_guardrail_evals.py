"""Unit tests for guardrail evaluation utilities."""

from __future__ import annotations

import os

import pytest

from guardrails.evals.core.types import Sample
from guardrails.evals.guardrail_evals import GuardrailEval


def _build_samples(count: int) -> list[Sample]:
    """Build synthetic samples for chunking tests.

    Args:
        count: Number of synthetic samples to build.

    Returns:
        List of Sample instances configured for evaluation.
    """
    return [Sample(id=f"sample-{idx}", data=f"payload-{idx}", expected_triggers={"g": bool(idx % 2)}) for idx in range(count)]


def test_determine_parallel_model_limit_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use cpu_count when explicit parallelism is not provided.

    Args:
        monkeypatch: Pytest monkeypatch helper.
    """
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    assert GuardrailEval._determine_parallel_model_limit(10, None) == 4
    assert GuardrailEval._determine_parallel_model_limit(2, None) == 2


def test_determine_parallel_model_limit_respects_request() -> None:
    """Honor user-provided parallelism constraints."""
    assert GuardrailEval._determine_parallel_model_limit(5, 3) == 3
    with pytest.raises(ValueError):
        GuardrailEval._determine_parallel_model_limit(5, 0)


def test_chunk_samples_without_size() -> None:
    """Return the original sample list when no chunk size is provided."""
    samples = _build_samples(3)
    chunks = list(GuardrailEval._chunk_samples(samples, None))
    assert len(chunks) == 1
    assert chunks[0] is samples


def test_chunk_samples_even_splits() -> None:
    """Split samples into evenly sized chunks."""
    samples = _build_samples(5)
    chunks = list(GuardrailEval._chunk_samples(samples, 2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [chunk[0].id for chunk in chunks] == ["sample-0", "sample-2", "sample-4"]


def test_chunk_samples_rejects_invalid_size() -> None:
    """Raise ValueError for non-positive chunk sizes."""
    samples = _build_samples(2)
    with pytest.raises(ValueError):
        list(GuardrailEval._chunk_samples(samples, 0))
