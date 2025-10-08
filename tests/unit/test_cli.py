"""Tests for guardrails CLI entry points."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from guardrails import cli


def _make_guardrail(media_type: str) -> Any:
    return SimpleNamespace(definition=SimpleNamespace(media_type=media_type))


def test_cli_validate_success(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Validate command should report total and matching guardrails."""

    class FakeStage:
        pass

    class FakePipeline:
        def __init__(self) -> None:
            self.pre_flight = FakeStage()
            self.input = FakeStage()
            self.output = FakeStage()

        def stages(self) -> list[FakeStage]:
            return [self.pre_flight, self.input, self.output]

    pipeline = FakePipeline()

    def fake_load_pipeline_bundles(path: Any) -> FakePipeline:
        assert str(path).endswith("config.json")  # noqa: S101
        return pipeline

    def fake_instantiate_guardrails(stage: Any, registry: Any | None = None) -> list[Any]:
        if stage is pipeline.pre_flight:
            return [_make_guardrail("text/plain")]
        if stage is pipeline.input:
            return [_make_guardrail("application/json")]
        if stage is pipeline.output:
            return [_make_guardrail("text/plain")]
        return []

    monkeypatch.setattr(cli, "load_pipeline_bundles", fake_load_pipeline_bundles)
    monkeypatch.setattr(cli, "instantiate_guardrails", fake_instantiate_guardrails)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["validate", "config.json", "--media-type", "text/plain"])

    assert excinfo.value.code == 0  # noqa: S101
    stdout = capsys.readouterr().out
    assert "Config valid" in stdout  # noqa: S101
    assert "2 matching media-type 'text/plain'" in stdout  # noqa: S101


def test_cli_validate_handles_errors(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Validation errors should print to stderr and exit with status 1."""

    def fake_load_pipeline_bundles(path: Any) -> None:
        raise ValueError("failed to load")

    monkeypatch.setattr(cli, "load_pipeline_bundles", fake_load_pipeline_bundles)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["validate", "bad.json"])

    assert excinfo.value.code == 1  # noqa: S101
    stderr = capsys.readouterr().err
    assert "ERROR: failed to load" in stderr  # noqa: S101
