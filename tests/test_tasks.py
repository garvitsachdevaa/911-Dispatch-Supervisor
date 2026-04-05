"""Tests for task registry and scenario fixtures (dispatch domain)."""

from __future__ import annotations

import pytest

from src.tasks.registry import DispatchScenarioFactory, TaskRegistry


def test_four_tasks_registered() -> None:
    tasks = TaskRegistry.list_tasks()
    ids = {t.task_id for t in tasks}
    assert ids == {"single_incident", "multi_incident", "mass_casualty", "shift_surge"}


def test_get_unknown_task_raises() -> None:
    with pytest.raises(KeyError):
        TaskRegistry.get("nope")


def test_factory_build_returns_state_and_meta() -> None:
    state, meta = DispatchScenarioFactory.build("multi_incident", seed=42)
    assert "units" in state
    assert "incidents" in state
    assert "task_id" in state
    assert meta["grid_size"]


def test_factory_is_deterministic() -> None:
    s1, m1 = DispatchScenarioFactory.build("shift_surge", seed=123)
    s2, m2 = DispatchScenarioFactory.build("shift_surge", seed=123)
    assert s1 == s2
    assert m1 == m2
