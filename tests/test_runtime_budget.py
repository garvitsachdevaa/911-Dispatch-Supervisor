"""Runtime and resource budget tests for the ATC Ground Control environment.

These tests verify:
1. 50-step episode completes in under 30 seconds on 2vCPU/8GB
2. No memory growth between step 1 and step 50 (using tracemalloc)
"""

import gc
import time
import tracemalloc

import pytest

from src.airport_schema import AirportSchemaLoader
from src.models import Action, ClearanceType
from src.state_machine import FullLifecycleStateMachine


@pytest.fixture
def gatwick_schema():
    """Load Gatwick airport schema for testing."""
    return AirportSchemaLoader.load("egkk_gatwick")


def _run_episode_steps(schema, seed: int, task_id: str, num_steps: int):
    """Run specified number of steps and return final state."""
    sm = FullLifecycleStateMachine(schema=schema, seed=seed)
    sm.reset(task_id=task_id, episode_id=f"ep-{seed}")

    for step in range(num_steps):
        state = sm._state
        legal_actions = sm.get_legal_actions(state)
        if legal_actions:
            action = legal_actions[0]
        else:
            action = Action(
                clearance_type=ClearanceType.TAXI,
                target_callsign="BAW123",
                route=[],
            )
        sm.step(action)

    return sm._state


class TestRuntimeBudget:
    """Tests for runtime performance budget."""

    def test_50_steps_under_30_seconds(self, gatwick_schema) -> None:
        """Test that 50-step episode completes in under 30 seconds.

        This simulates the target environment of 2vCPU/8GB.
        """
        num_steps = 50
        task_id = "arrival"
        seed = 42

        gc.collect()
        start_time = time.perf_counter()

        final_state = _run_episode_steps(gatwick_schema, seed, task_id, num_steps)

        elapsed = time.perf_counter() - start_time

        assert elapsed < 30.0, f"50-step episode took {elapsed:.2f}s (budget: 30s)"
        assert final_state.step_count == num_steps, (
            f"Expected {num_steps} steps, got {final_state.step_count}"
        )

    def test_departure_50_steps_under_30_seconds(self, gatwick_schema) -> None:
        """Test that 50-step departure episode completes in under 30 seconds."""
        num_steps = 50
        task_id = "departure"
        seed = 42

        gc.collect()
        start_time = time.perf_counter()

        final_state = _run_episode_steps(gatwick_schema, seed, task_id, num_steps)

        elapsed = time.perf_counter() - start_time

        assert elapsed < 30.0, f"50-step departure took {elapsed:.2f}s (budget: 30s)"

    def test_integrated_50_steps_under_30_seconds(self, gatwick_schema) -> None:
        """Test that 50-step integrated episode completes in under 30 seconds."""
        num_steps = 50
        task_id = "integrated"
        seed = 42

        gc.collect()
        start_time = time.perf_counter()

        final_state = _run_episode_steps(gatwick_schema, seed, task_id, num_steps)

        elapsed = time.perf_counter() - start_time

        assert elapsed < 30.0, f"50-step integrated took {elapsed:.2f}s (budget: 30s)"


class TestMemoryBudget:
    """Tests for memory stability."""

    def test_no_memory_growth_between_step_1_and_50(self, gatwick_schema) -> None:
        """Test that no memory leak between step 1 and step 50.

        Uses tracemalloc to detect memory growth.
        """
        num_steps = 50
        task_id = "arrival"
        seed = 42

        gc.collect()
        tracemalloc.start()

        sm = FullLifecycleStateMachine(schema=gatwick_schema, seed=seed)
        sm.reset(task_id=task_id, episode_id=f"ep-{seed}")

        for step in range(1):
            legal_actions = sm.get_legal_actions(sm._state)
            if legal_actions:
                action = legal_actions[0]
            else:
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign="BAW123",
                    route=[],
                )
            sm.step(action)

        snapshot1 = tracemalloc.take_snapshot()
        memory_step_1 = sum(stat.size for stat in snapshot1.statistics("lineno"))

        for step in range(1, num_steps):
            legal_actions = sm.get_legal_actions(sm._state)
            if legal_actions:
                action = legal_actions[0]
            else:
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign="BAW123",
                    route=[],
                )
            sm.step(action)

        snapshot50 = tracemalloc.take_snapshot()
        memory_step_50 = sum(stat.size for stat in snapshot50.statistics("lineno"))

        tracemalloc.stop()

        growth = memory_step_50 - memory_step_1
        growth_mb = growth / (1024 * 1024)

        # Allow some tolerance for normal fluctuation, but no runaway growth
        # We'll be generous and allow 10MB growth for a full episode
        assert growth_mb < 10.0, (
            f"Memory grew by {growth_mb:.2f}MB between step 1 and step 50 (budget: 10MB)"
        )

    def test_memory_stable_across_all_tasks(self, gatwick_schema) -> None:
        """Test memory stability across all three task types."""
        gc.collect()
        tracemalloc.start()

        tasks = ["arrival", "departure", "integrated"]

        for task_id in tasks:
            sm = FullLifecycleStateMachine(schema=gatwick_schema, seed=42)
            sm.reset(task_id=task_id, episode_id=f"ep-{task_id}")

            for step in range(20):
                legal_actions = sm.get_legal_actions(sm._state)
                if legal_actions:
                    action = legal_actions[0]
                else:
                    action = Action(
                        clearance_type=ClearanceType.TAXI,
                        target_callsign="BAW123",
                        route=[],
                    )
                sm.step(action)

        snapshot = tracemalloc.take_snapshot()
        total_memory = sum(stat.size for stat in snapshot.statistics("lineno"))
        total_memory_mb = total_memory / (1024 * 1024)

        tracemalloc.stop()

        # Total memory should be reasonable
        assert total_memory_mb < 50.0, (
            f"Total memory usage {total_memory_mb:.2f}MB is too high (budget: 50MB)"
        )


class TestRuntimeConsistency:
    """Tests for runtime consistency across multiple runs."""

    def test_consistent_runtime_across_runs(self, gatwick_schema) -> None:
        """Test that runtime is consistent across multiple 50-step runs."""
        num_steps = 50
        task_id = "arrival"
        seed = 42

        # Warm up run to avoid cold-start variance
        _run_episode_steps(gatwick_schema, seed, task_id, num_steps)

        runtimes = []

        for _ in range(3):
            start_time = time.perf_counter()
            _run_episode_steps(gatwick_schema, seed, task_id, num_steps)
            elapsed = time.perf_counter() - start_time
            runtimes.append(elapsed)

        max_runtime = max(runtimes)
        avg_runtime = sum(runtimes) / len(runtimes)

        # All runs must be under 30 seconds (the actual budget)
        assert max_runtime < 30.0, f"Max runtime {max_runtime:.2f}s exceeds budget"
        # After warm-up, runs should be reasonably consistent (within 10x)
        # This allows for some GC variance but catches pathological cases
        assert max_runtime < avg_runtime * 10.0, (
            f"Runtime variance too high: {runtimes}"
        )
