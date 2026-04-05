"""Dispatch task test: Single Incident Response."""

from __future__ import annotations

from src.city_schema import CitySchemaLoader
from src.models import Action, DispatchAction
from src.tasks.single_incident import SingleIncidentGrader, SingleIncidentTask


def test_single_incident_task_reset_and_step() -> None:
    schema = CitySchemaLoader.load("metro_city")
    task = SingleIncidentTask(city_schema=schema, seed=42)
    state = task.reset(episode_id="ep-1")
    assert state.task_id == "single_incident"
    assert state.step_count == 0
    assert state.units
    assert state.incidents

    legal = task.state_machine.get_legal_actions(state)
    assert legal

    next_state, obs = task.step(state, legal[0])
    assert next_state.step_count == 1
    assert obs.protocol_ok is True


def test_single_incident_grader_in_range() -> None:
    schema = CitySchemaLoader.load("metro_city")
    task = SingleIncidentTask(city_schema=schema, seed=42)
    state = task.reset(episode_id="ep-1")

    action = Action(
        action_type=DispatchAction.DISPATCH,
        unit_id="MED-1",
        incident_id="INC-001",
    )
    state, obs = task.step(state, action)
    grader = SingleIncidentGrader()
    score = grader.grade(state, [0.0, obs.score])
    assert 0.0 <= score <= 1.0
