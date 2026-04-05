"""Tests for task registry and scenario fixtures."""

import pytest

from src.tasks.registry import (
    ScenarioFixtureFactory,
    TaskInfo,
    TaskRegistry,
)


class TestTaskRegistry:
    def test_exactly_four_tasks_registered(self):
        tasks = TaskRegistry.list_tasks()
        assert len(tasks) == 4

    def test_departure_task_is_easy(self):
        task = TaskRegistry.get("departure")
        assert task.difficulty == "easy"

    def test_arrival_task_is_hard(self):
        task = TaskRegistry.get("arrival")
        assert task.difficulty == "hard"

    def test_integrated_task_is_medium(self):
        task = TaskRegistry.get("integrated")
        assert task.difficulty == "medium"

    def test_get_unknown_task_raises_keyerror(self):
        with pytest.raises(KeyError):
            TaskRegistry.get("nonexistent")

    def test_list_by_difficulty_easy(self):
        easy_tasks = TaskRegistry.list_by_difficulty("easy")
        assert len(easy_tasks) == 1
        assert easy_tasks[0].task_id == "departure"

    def test_list_by_difficulty_hard(self):
        hard_tasks = TaskRegistry.list_by_difficulty("hard")
        assert len(hard_tasks) == 2
        hard_ids = [t.task_id for t in hard_tasks]
        assert "arrival" in hard_ids
        assert "peak_traffic" in hard_ids

    def test_list_by_difficulty_medium(self):
        medium_tasks = TaskRegistry.list_by_difficulty("medium")
        assert len(medium_tasks) == 1
        assert medium_tasks[0].task_id == "integrated"

    def test_list_by_difficulty_no_results(self):
        tasks = TaskRegistry.list_by_difficulty("impossible")
        assert len(tasks) == 0

    def test_task_ids_are_unique(self):
        task_ids = [t.task_id for t in TaskRegistry.list_tasks()]
        assert len(task_ids) == len(set(task_ids))


class TestScenarioFixtureFactory:
    def test_build_unknown_task_raises_keyerror(self):
        with pytest.raises(KeyError):
            ScenarioFixtureFactory.build("unknown_task", seed=42)

    def test_seed_determinism_departure(self):
        state1, actions1 = ScenarioFixtureFactory.build_departure_fixture(seed=12345)
        state2, actions2 = ScenarioFixtureFactory.build_departure_fixture(seed=12345)
        assert state1 == state2
        assert actions1 == actions2

    def test_seed_determinism_arrival(self):
        state1, actions1 = ScenarioFixtureFactory.build_arrival_fixture(seed=12345)
        state2, actions2 = ScenarioFixtureFactory.build_arrival_fixture(seed=12345)
        assert state1 == state2
        assert actions1 == actions2

    def test_seed_determinism_integrated(self):
        state1, actions1 = ScenarioFixtureFactory.build_integrated_fixture(seed=12345)
        state2, actions2 = ScenarioFixtureFactory.build_integrated_fixture(seed=12345)
        assert state1 == state2
        assert actions1 == actions2

    def test_different_seeds_produce_different_departure_fixtures(self):
        state1, actions1 = ScenarioFixtureFactory.build_departure_fixture(seed=111)
        state2, actions2 = ScenarioFixtureFactory.build_departure_fixture(seed=222)
        assert state1 != state2 or actions1 != actions2

    def test_different_seeds_produce_different_arrival_fixtures(self):
        state1, actions1 = ScenarioFixtureFactory.build_arrival_fixture(seed=111)
        state2, actions2 = ScenarioFixtureFactory.build_arrival_fixture(seed=222)
        assert state1 != state2 or actions1 != actions2

    def test_different_seeds_produce_different_integrated_fixtures(self):
        state1, actions1 = ScenarioFixtureFactory.build_integrated_fixture(seed=111)
        state2, actions2 = ScenarioFixtureFactory.build_integrated_fixture(seed=222)
        assert state1 != state2 or actions1 != actions2

    def test_build_departure_fixture_returns_expected_keys(self):
        state, actions = ScenarioFixtureFactory.build_departure_fixture(seed=42)
        assert "phase" in state
        assert "aircraft" in state
        assert "episode_id" in state
        assert "task_id" in state
        assert state["task_id"] == "departure"
        assert isinstance(actions, list)

    def test_build_arrival_fixture_returns_expected_keys(self):
        state, actions = ScenarioFixtureFactory.build_arrival_fixture(seed=42)
        assert "phase" in state
        assert "aircraft" in state
        assert "episode_id" in state
        assert "task_id" in state
        assert state["task_id"] == "arrival"
        assert isinstance(actions, list)

    def test_build_integrated_fixture_returns_expected_keys(self):
        state, actions = ScenarioFixtureFactory.build_integrated_fixture(seed=42)
        assert "phase" in state
        assert "aircraft" in state
        assert "episode_id" in state
        assert "task_id" in state
        assert state["task_id"] == "integrated"
        assert isinstance(actions, list)

    def test_build_method_routes_correctly_to_departure(self):
        state, actions = ScenarioFixtureFactory.build("departure", seed=42)
        assert state["task_id"] == "departure"
        assert len(actions) > 0

    def test_build_method_routes_correctly_to_arrival(self):
        state, actions = ScenarioFixtureFactory.build("arrival", seed=42)
        assert state["task_id"] == "arrival"
        assert len(actions) > 0

    def test_build_method_routes_correctly_to_integrated(self):
        state, actions = ScenarioFixtureFactory.build("integrated", seed=42)
        assert state["task_id"] == "integrated"
        assert len(actions) > 0

    def test_departure_fixture_has_aircraft_callsign(self):
        state, _ = ScenarioFixtureFactory.build_departure_fixture(seed=42)
        aircraft = state["aircraft"]
        assert len(aircraft) == 1
        callsign = list(aircraft.keys())[0]
        assert len(callsign) >= 3
        assert callsign[:3].isalpha()

    def test_arrival_fixture_has_aircraft_callsign(self):
        state, _ = ScenarioFixtureFactory.build_arrival_fixture(seed=42)
        aircraft = state["aircraft"]
        assert len(aircraft) == 1
        callsign = list(aircraft.keys())[0]
        assert len(callsign) >= 3
        assert callsign[:3].isalpha()

    def test_integrated_fixture_has_aircraft_callsign(self):
        state, _ = ScenarioFixtureFactory.build_integrated_fixture(seed=42)
        aircraft = state["aircraft"]
        assert len(aircraft) == 1
        callsign = list(aircraft.keys())[0]
        assert len(callsign) >= 3
        assert callsign[:3].isalpha()


class TestTaskInfoModel:
    def test_task_info_extra_forbid(self):
        with pytest.raises(Exception):
            TaskInfo(
                task_id="test",
                name="Test",
                description="Test desc",
                difficulty="easy",
                initial_state_fn="fn",
                extra_field="should_fail",
            )

    def test_task_info_valid_construction(self):
        task = TaskInfo(
            task_id="test",
            name="Test Task",
            description="A test task",
            difficulty="easy",
            initial_state_fn="build_test_fixture",
        )
        assert task.task_id == "test"
        assert task.difficulty == "easy"
