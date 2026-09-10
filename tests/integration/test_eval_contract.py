"""Exercise the public local-check evaluation workflow from JSONL to reports."""

import json
from pathlib import Path
from typing import Any

import pytest

from guardrails.evals import GuardrailEval, JsonlDatasetLoader, validate_dataset


@pytest.fixture
def evaluation_files(tmp_path: Path) -> tuple[Path, Path]:
    bundle = {"version": 1, "guardrails": [{"name": "Keyword Filter", "config": {"keywords": ["blocked"]}}]}
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"version": 1, "pre_flight": bundle, "output": bundle}))
    dataset = tmp_path / "samples.jsonl"
    # One example in each confusion-matrix cell provides an independent oracle.
    rows = [
        {"id": "tp", "data": "blocked", "expected_triggers": {"Keyword Filter": True}},
        {"id": "fp", "data": "blocked", "expected_triggers": {"Keyword Filter": False}},
        {"id": "fn", "data": "allowed", "expected_triggers": {"Keyword Filter": True}},
        {"id": "tn", "data": "allowed", "expected_triggers": {"Keyword Filter": False}},
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return config, dataset


@pytest.mark.asyncio
@pytest.mark.parametrize("stages", [None, ["output"]])
async def test_evaluation_persists_correct_results_and_metrics(evaluation_files: tuple[Path, Path], tmp_path: Path, stages: list[str] | None) -> None:
    config, dataset = evaluation_files
    output = tmp_path / "reports"
    evaluator = GuardrailEval(config, dataset, stages=stages, batch_size=2, output_dir=output, api_key="test-key")
    await evaluator.run()

    [run_dir] = list(output.iterdir())
    metrics = json.loads((run_dir / "eval_metrics.json").read_text())
    expected_stages = {"pre_flight", "output"} if stages is None else {"output"}
    assert set(metrics) == expected_stages
    for stage in expected_stages:
        assert metrics[stage] == {
            "Keyword Filter": {
                "true_positives": 1,
                "false_positives": 1,
                "false_negatives": 1,
                "true_negatives": 1,
                "total_samples": 4,
                "precision": 0.5,
                "recall": 0.5,
                "f1_score": 0.5,
            }
        }
        rows = [json.loads(line) for line in (run_dir / f"eval_results_{stage}.jsonl").read_text().splitlines()]
        assert len(rows) == 4
        assert {row["id"]: row["triggered"]["Keyword Filter"] for row in rows} == {"tp": True, "fp": True, "fn": False, "tn": False}
        assert {row["id"]: row["expected_triggers"]["Keyword Filter"] for row in rows} == {"tp": True, "fp": False, "fn": True, "tn": False}
    assert "Total samples: 4" in (run_dir / "run_summary.txt").read_text()
    assert {path.name for path in run_dir.glob("eval_results_*.jsonl")} == {f"eval_results_{stage}.jsonl" for stage in expected_stages}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["invalid_dataset", "unknown_check", "missing_stage"])
async def test_invalid_evaluation_does_not_write_success_report(evaluation_files: tuple[Path, Path], tmp_path: Path, failure: str) -> None:
    config, dataset = evaluation_files
    options: dict[str, Any] = {}
    if failure == "invalid_dataset":
        dataset.write_text('{"id": "missing-required-fields"}\n')
    elif failure == "unknown_check":
        config.write_text(json.dumps({"output": {"guardrails": [{"name": "not-registered", "config": {}}]}}))
    else:
        options["stages"] = ["input"]
    output = tmp_path / "reports"
    evaluator = GuardrailEval(config, dataset, output_dir=output, api_key="test-key", **options)
    with pytest.raises(ValueError):
        await evaluator.run()
    assert not output.exists()


def test_dataset_validation_reports_line_and_loader_rejects_invalid_sample(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"id":"ok","data":"text","expected_triggers":{}}\nnot json\n')
    valid, messages = validate_dataset(path)
    assert valid is False
    assert any("Line 2" in message for message in messages)
    with pytest.raises(ValueError, match="line 2"):
        JsonlDatasetLoader().load(path)
