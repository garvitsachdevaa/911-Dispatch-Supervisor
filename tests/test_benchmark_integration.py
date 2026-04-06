"""Integration tests for benchmark assembly and score ranges."""

from __future__ import annotations

from src.benchmark import list_tasks, run_all, run_task


def test_list_tasks_has_four() -> None:
    tasks = list_tasks()
    assert len(tasks) == 4
    ids = {t["task_id"] for t in tasks}
    assert ids == {"single_incident", "multi_incident", "mass_casualty", "shift_surge"}


def test_run_task_score_in_range() -> None:
    result = run_task("single_incident", seed=42)
    assert 0.0 <= result["score"] <= 1.0
    assert result["task_id"] == "single_incident"
    # Benchmark scoring must match the OpenEnv evaluation path: mean step reward.
    rewards = result["rewards"]
    if rewards:
        expected = sum(rewards) / len(rewards)
    else:
        expected = 0.0
    assert abs(result["score"] - expected) < 1e-9


def test_run_all_scores_in_range() -> None:
    scores = run_all()
    assert set(scores.keys()) == {"single_incident", "multi_incident", "mass_casualty", "shift_surge"}
    assert all(0.0 <= s <= 1.0 for s in scores.values())
