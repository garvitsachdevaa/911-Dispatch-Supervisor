"""Determinism tests for 911 dispatch fixtures and environment."""

from __future__ import annotations

import asyncio

from src.openenv_environment import OpenEnvEnvironment
from src.tasks.registry import DispatchScenarioFactory


def test_fixture_determinism_same_seed() -> None:
    s1, m1 = DispatchScenarioFactory.build("single_incident", seed=123)
    s2, m2 = DispatchScenarioFactory.build("single_incident", seed=123)
    assert s1 == s2
    assert m1 == m2


def test_fixture_variation_different_seed() -> None:
    s1, _ = DispatchScenarioFactory.build("single_incident", seed=111)
    s2, _ = DispatchScenarioFactory.build("single_incident", seed=222)
    assert s1 != s2


def test_openenv_same_seed_same_initial_units_and_incidents() -> None:
    env1 = OpenEnvEnvironment(task_id="multi_incident", seed=42)
    env2 = OpenEnvEnvironment(task_id="multi_incident", seed=42)

    asyncio.run(env1.reset())
    asyncio.run(env2.reset())

    st1 = env1.state().model_dump()
    st2 = env2.state().model_dump()

    st1.pop("episode_id", None)
    st2.pop("episode_id", None)
    assert st1["units"] == st2["units"]
    assert st1["incidents"] == st2["incidents"]

    env1.close()
    env2.close()
