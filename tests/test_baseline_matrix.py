"""Unit tests for scripts/run_baseline_matrix.py helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_baseline_matrix.py"
SPEC = importlib.util.spec_from_file_location("run_baseline_matrix", SCRIPT_PATH)
assert SPEC and SPEC.loader
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


def test_extract_task_episodes_parses_start_end_pairs() -> None:
    stdout = "\n".join(
        [
            "[START] task=single_incident env=citywide-dispatch-supervisor model=test-model",
            "[STEP] step=1 action=WAIT reward=0.00 done=false error=null",
            "[END] success=true steps=20 score=0.300 rewards=0.00,0.10",
            "[START] task=multi_incident env=citywide-dispatch-supervisor model=test-model",
            "[END] success=true steps=40 score=0.700 rewards=0.10,0.20",
        ]
    )

    episodes = baseline._extract_task_episodes(stdout)

    assert len(episodes) == 2
    assert episodes[0].task_id == "single_incident"
    assert episodes[0].success is True
    assert episodes[0].steps == 20
    assert episodes[0].score == pytest.approx(0.3)
    assert episodes[1].task_id == "multi_incident"
    assert episodes[1].steps == 40
    assert episodes[1].score == pytest.approx(0.7)


def test_extract_task_episodes_falls_back_to_unknown_task() -> None:
    stdout = "[END] success=false steps=0 score=0.000 rewards=0.00"

    episodes = baseline._extract_task_episodes(stdout)

    assert len(episodes) == 1
    assert episodes[0].task_id == "unknown-1"
    assert episodes[0].success is False


def test_summarize_computes_mean_and_std() -> None:
    runs = [
        baseline.RunResult(
            lane="random",
            run_index=1,
            runtime_seconds=1.0,
            tasks=[baseline.TaskEpisode("single_incident", True, 20, 0.2)],
            return_code=0,
            stderr="",
        ),
        baseline.RunResult(
            lane="random",
            run_index=2,
            runtime_seconds=1.1,
            tasks=[baseline.TaskEpisode("single_incident", True, 20, 0.4)],
            return_code=0,
            stderr="",
        ),
    ]

    summary = baseline._summarize(runs)

    assert summary["single_incident"]["runs"] == 2.0
    assert summary["single_incident"]["mean"] == pytest.approx(0.3)
    assert summary["single_incident"]["std"] == pytest.approx(0.1)
    assert summary["single_incident"]["min"] == pytest.approx(0.2)
    assert summary["single_incident"]["max"] == pytest.approx(0.4)


def test_to_jsonable_serializes_runs() -> None:
    runs = [
        baseline.RunResult(
            lane="llm",
            run_index=1,
            runtime_seconds=3.2,
            tasks=[baseline.TaskEpisode("mass_casualty", True, 59, 0.742)],
            return_code=0,
            stderr="",
        )
    ]

    payload = baseline._to_jsonable(runs)

    assert payload[0]["lane"] == "llm"
    assert payload[0]["tasks"][0]["task_id"] == "mass_casualty"
    assert payload[0]["tasks"][0]["score"] == pytest.approx(0.742)
