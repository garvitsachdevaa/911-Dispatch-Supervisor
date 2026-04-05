"""Dispatch task test: Mass Casualty."""

from __future__ import annotations

from src.city_schema import CitySchemaLoader
from src.tasks.mass_casualty import MassCasualtyGrader, MassCasualtyTask


def test_mass_casualty_task_spawns_waves() -> None:
    schema = CitySchemaLoader.load("metro_city")
    task = MassCasualtyTask(city_schema=schema, seed=42)
    state = task.reset(episode_id="ep-1")

    assert state.task_id == "mass_casualty"
    assert state.incidents
    assert state.metadata.get("waves") is not None

    legal = task.state_machine.get_legal_actions(state)
    if legal:
        state, obs = task.step(state, legal[0])
        assert obs.score >= 0.0


def test_mass_casualty_grader_in_range() -> None:
    schema = CitySchemaLoader.load("metro_city")
    task = MassCasualtyTask(city_schema=schema, seed=42)
    state = task.reset(episode_id="ep-1")
    grader = MassCasualtyGrader()
    score = grader.grade(state, [0.0, 0.3, 0.6])
    assert 0.0 <= score <= 1.0
