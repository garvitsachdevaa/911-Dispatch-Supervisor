"""Integration tests for benchmark assembly and grader score validation."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from src.airport_schema import AirportSchemaLoader
from src.benchmark import list_tasks, run_task, run_all
from src.models import Action, ClearanceType, LifecyclePhase
from src.state_machine import FullLifecycleStateMachine, LifecycleState
from src.tasks.arrival import ArrivalGrader
from src.tasks.departure import DepartureGrader
from src.tasks.integrated import IntegratedGrader
from src.tasks.registry import ScenarioFixtureFactory


class TestGraderScoreRanges:
    """Verify all graders return scores in [0.0, 1.0]."""

    def test_arrival_grader_returns_score_in_range(self):
        grader = ArrivalGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="arrival", episode_id="test-arrival")
        rewards = [0.5, 0.6, 0.7, 0.8]

        score = grader.grade(state, rewards)
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0.0, 1.0]"

    def test_arrival_grader_handles_empty_rewards(self):
        grader = ArrivalGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="arrival", episode_id="test-arrival")
        score = grader.grade(state, [])
        assert score == 0.0

    def test_departure_grader_returns_score_in_range(self):
        grader = DepartureGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="departure", episode_id="test-departure")
        rewards = [0.4, 0.5, 0.6, 0.7]

        score = grader.grade(state, rewards)
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0.0, 1.0]"

    def test_departure_grader_handles_empty_rewards(self):
        grader = DepartureGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="departure", episode_id="test-departure")
        score = grader.grade(state, [])
        assert score == 0.0

    def test_integrated_grader_returns_score_in_range(self):
        grader = IntegratedGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="integrated", episode_id="test-integrated")
        rewards = [0.3, 0.4, 0.5, 0.6, 0.7]

        score = grader.grade(state, rewards)
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0.0, 1.0]"

    def test_integrated_grader_handles_empty_rewards(self):
        grader = IntegratedGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="integrated", episode_id="test-integrated")
        score = grader.grade(state, [])
        assert score == 0.0


class TestBenchmarkInterface:
    """Test benchmark module interface."""

    def test_list_tasks_returns_four_tasks(self):
        tasks = list_tasks()
        assert len(tasks) == 4

        task_ids = [t["task_id"] for t in tasks]
        assert "arrival" in task_ids
        assert "departure" in task_ids
        assert "integrated" in task_ids
        assert "peak_traffic" in task_ids

    def test_list_tasks_returns_correct_metadata(self):
        tasks = list_tasks()
        for task in tasks:
            assert "task_id" in task
            assert "name" in task
            assert "difficulty" in task
            assert task["difficulty"] in ("easy", "medium", "hard")

    def test_run_task_returns_score_in_range(self):
        result = run_task("arrival", seed=42)
        assert "score" in result
        assert 0.0 <= result["score"] <= 1.0
        assert "task_id" in result
        assert result["task_id"] == "arrival"

    def test_run_task_departure_score_in_range(self):
        result = run_task("departure", seed=42)
        assert 0.0 <= result["score"] <= 1.0

    def test_run_task_integrated_score_in_range(self):
        result = run_task("integrated", seed=42)
        assert 0.0 <= result["score"] <= 1.0

    def test_run_task_invalid_raises_keyerror(self):
        with pytest.raises(KeyError):
            run_task("nonexistent", seed=42)

    def test_run_all_returns_all_scores_in_range(self):
        scores = run_all()

        assert len(scores) == 4
        assert "arrival" in scores
        assert "departure" in scores
        assert "integrated" in scores
        assert "peak_traffic" in scores

        for task_id, score in scores.items():
            assert 0.0 <= score <= 1.0, (
                f"Score {score} for {task_id} outside [0.0, 1.0]"
            )


class TestScenarioFixtureGrading:
    """Test grading with scenario fixtures."""

    def test_arrival_fixture_through_grader(self):
        grader = ArrivalGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="arrival", episode_id="test-arrival-fixture")

        rewards = []
        for _ in range(10):
            state, obs = sm.step(
                Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign="BAW123",
                    runway="27L",
                )
            )
            rewards.append(obs.score)

        score = grader.grade(state, rewards)
        assert 0.0 <= score <= 1.0

    def test_departure_fixture_through_grader(self):
        grader = DepartureGrader()
        schema = AirportSchemaLoader.load("dummy_small")
        sm = FullLifecycleStateMachine(schema=schema, seed=42)

        state = sm.reset(task_id="departure", episode_id="test-departure-fixture")

        rewards = []
        for _ in range(10):
            state, obs = sm.step(
                Action(
                    clearance_type=ClearanceType.PUSHBACK,
                    target_callsign="BAW123",
                    pushback_direction="back",
                )
            )
            rewards.append(obs.score)

        score = grader.grade(state, rewards)
        assert 0.0 <= score <= 1.0


class TestPrevalidationChecks:
    """Test prevalidation checks (docker availability)."""

    def test_docker_build_available(self):
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            docker_available = False

        if docker_available:
            build_result = subprocess.run(
                ["docker", "build", "-t", "gradient-ascent-atc-test", "."],
                capture_output=True,
                timeout=300,
            )
            assert build_result.returncode == 0, "Docker build failed"
        else:
            pytest.skip("Docker not available")
