"""Dispatch task test: Multi Incident."""

from __future__ import annotations

from src.city_schema import CitySchemaLoader
from src.tasks.multi_incident import MultiIncidentGrader, MultiIncidentTask


def test_multi_incident_task_smoke() -> None:
    schema = CitySchemaLoader.load("metro_city")
    task = MultiIncidentTask(city_schema=schema, seed=42)
    state = task.reset(episode_id="ep-1")
    assert state.task_id == "multi_incident"
    assert len(state.incidents) >= 3

    legal = task.state_machine.get_legal_actions(state)
    assert legal
    state, obs = task.step(state, legal[0])
    assert obs.result in {"ok", "invalid action"}


def test_multi_incident_grader_in_range() -> None:
    schema = CitySchemaLoader.load("metro_city")
    task = MultiIncidentTask(city_schema=schema, seed=42)
    state = task.reset(episode_id="ep-1")
    grader = MultiIncidentGrader()
    score = grader.grade(state, [0.0, 0.5, 0.8])
    assert 0.0 <= score <= 1.0
