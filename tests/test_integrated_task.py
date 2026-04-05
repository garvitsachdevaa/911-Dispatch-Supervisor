"""Tests for integrated full-lifecycle task."""

import pytest

from src.models import Action, ClearanceType, LifecyclePhase, Observation
from src.rewards import RewardCalculator
from src.state_machine import FullLifecycleStateMachine, LifecycleState
from src.tasks.integrated import IntegratedGrader, IntegratedTask
from src.airport_schema import AirportSchemaLoader


class TestIntegratedTask:
    """Tests for IntegratedTask class."""

    @pytest.fixture
    def integrated_task(self) -> IntegratedTask:
        """Create an IntegratedTask instance."""
        return IntegratedTask()

    def test_integrated_task_instantiation(
        self, integrated_task: IntegratedTask
    ) -> None:
        """Test that IntegratedTask can be instantiated."""
        assert integrated_task is not None

    def test_turnaround_delay_default(self, integrated_task: IntegratedTask) -> None:
        """Test that turnaround delay defaults to 60 seconds."""
        assert hasattr(integrated_task, "turnaround_delay_s")
        assert integrated_task.turnaround_delay_s == 60.0


class TestIntegratedGrader:
    """Tests for IntegratedGrader class."""

    @pytest.fixture
    def grader(self) -> IntegratedGrader:
        """Create an IntegratedGrader instance."""
        return IntegratedGrader()

    @pytest.fixture
    def calculator(self) -> RewardCalculator:
        """Create a RewardCalculator instance."""
        return RewardCalculator()

    def test_grader_instantiation(self, grader: IntegratedGrader) -> None:
        """Test that IntegratedGrader can be instantiated."""
        assert grader is not None

    def test_grade_returns_float(
        self, grader: IntegratedGrader, calculator: RewardCalculator
    ) -> None:
        """Test that grade returns a float."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test-123",
            step_count=11,
            task_id="integrated",
            completed_phases=list(LifecyclePhase),
            metadata={},
        )
        rewards = [0.8] * 11
        result = grader.grade(state, rewards)
        assert isinstance(result, float)

    def test_grade_in_range_zero_to_one(
        self, grader: IntegratedGrader, calculator: RewardCalculator
    ) -> None:
        """Test that grade is clamped to [0.0, 1.0]."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test-123",
            step_count=11,
            task_id="integrated",
            completed_phases=list(LifecyclePhase),
            metadata={},
        )
        rewards = [1.0] * 20
        result = grader.grade(state, rewards)
        assert 0.0 <= result <= 1.0

    def test_full_lifecycle_success_bonus(
        self, grader: IntegratedGrader, calculator: RewardCalculator
    ) -> None:
        """Test that completing all 11 phases safely grants +0.2 bonus."""
        # All 11 phases completed safely
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test-123",
            step_count=15,
            task_id="integrated",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={},
        )
        # All steps get perfect rewards
        rewards = [1.0] * 15
        result = grader.grade(state, rewards)
        # Should include full completion bonus
        assert result > 0.8  # Base score + 0.2 bonus

    def test_phase_failure_reduces_score(
        self, grader: IntegratedGrader, calculator: RewardCalculator
    ) -> None:
        """Test that each missed/unsafe phase reduces score by 0.05."""
        # Missing 2 phases (e.g., taxi_in and docking failed)
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test-123",
            step_count=10,
            task_id="integrated",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                # TAXI_IN and DOCKING failed
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={},
        )
        rewards = [0.8] * 10
        result = grader.grade(state, rewards)
        # Missing 2 phases = 0.1 penalty
        # Base 0.8 with 0.1 penalty should give ~0.7
        assert result < 0.8

    def test_score_range_all_phases_completed(
        self, grader: IntegratedGrader, calculator: RewardCalculator
    ) -> None:
        """Test score range when all phases completed."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test-123",
            step_count=11,
            task_id="integrated",
            completed_phases=list(LifecyclePhase),
            metadata={},
        )
        rewards = [0.5] * 11
        result = grader.grade(state, rewards)
        # Should be clamped to [0.0, 1.0]
        assert 0.0 <= result <= 1.0


class TestIntegratedFullLifecycle:
    """Integration tests for full lifecycle."""

    @pytest.fixture
    def schema(self) -> AirportSchemaLoader:
        """Load the default airport schema."""
        return AirportSchemaLoader.load("egkk_gatwick")

    def test_full_lifecycle_episode_completes(
        self, schema: AirportSchemaLoader
    ) -> None:
        """Test that a full lifecycle episode can complete through docking.

        Note: Full lifecycle to DEPARTED requires unrealistic taxi distances
        in the current schema. This test verifies the state machine can
        progress through the first 5 phases (approach -> docking).
        """
        sm = FullLifecycleStateMachine(schema=schema, seed=42)
        state = sm.reset(task_id="integrated", episode_id="test-full-lifecycle")

        aircraft = list(state.aircraft_states.values())[0]
        callsign = aircraft.callsign
        # Override assigned_gate from fixture (GATE_C3) to a valid gate in schema
        aircraft.assigned_gate = "GATE_A1"

        target_phases = {
            LifecyclePhase.LANDING,
            LifecyclePhase.ARRIVAL_HANDOFF,
            LifecyclePhase.TAXI_IN,
            LifecyclePhase.DOCKING,
        }

        max_steps = 1000
        for step in range(max_steps):
            if sm.is_terminal(state):
                break

            legal_actions = sm.get_legal_actions(state)
            if legal_actions:
                action = legal_actions[0]
            else:
                action = Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=callsign,
                )

            state, obs = sm.step(action)

            if state.phase in target_phases:
                target_phases.discard(state.phase)

            if not target_phases:
                break

        assert (
            LifecyclePhase.DOCKING in state.completed_phases
            or state.phase == LifecyclePhase.DOCKING
        )
        assert LifecyclePhase.APPROACH in state.completed_phases
        assert LifecyclePhase.LANDING in state.completed_phases

    def test_phase_sequence_order(self, schema: AirportSchemaLoader) -> None:
        """Test that phases occur in the correct order."""
        sm = FullLifecycleStateMachine(schema=schema, seed=42)
        state = sm.reset(task_id="integrated", episode_id="test-phase-order")

        expected_order = [
            LifecyclePhase.APPROACH,
            LifecyclePhase.LANDING,
            LifecyclePhase.ARRIVAL_HANDOFF,
            LifecyclePhase.TAXI_IN,
            LifecyclePhase.DOCKING,
            LifecyclePhase.AT_GATE,
            LifecyclePhase.PUSHBACK,
            LifecyclePhase.TAXI_OUT,
            LifecyclePhase.DEPARTURE_QUEUE,
            LifecyclePhase.TAKEOFF,
            LifecyclePhase.DEPARTED,
        ]

        from src.tasks.registry import ScenarioFixtureFactory

        _, action_sequence = ScenarioFixtureFactory.build_integrated_fixture(seed=42)

        phases_reached = []
        step_index = 0
        max_steps = 1000
        for _ in range(max_steps):
            if sm.is_terminal(state):
                break

            if step_index < len(action_sequence):
                action_dict = action_sequence[step_index]
                action = Action(**action_dict)
            else:
                legal_actions = sm.get_legal_actions(state)
                if legal_actions:
                    action = legal_actions[0]
                else:
                    break

            prev_phase = state.phase
            state, obs = sm.step(action)
            if obs.result != "illegal_transition":
                step_index += 1
            if state.phase != prev_phase and state.phase not in phases_reached:
                phases_reached.append(state.phase)

        for i in range(len(phases_reached) - 1):
            try:
                current_idx = expected_order.index(phases_reached[i])
                next_idx = expected_order.index(phases_reached[i + 1])
                assert current_idx < next_idx, f"Phase order violated"
            except ValueError:
                pass

    def test_turnaround_delay_after_at_gate(self, schema: AirportSchemaLoader) -> None:
        """Test that turnaround delay is enforced in the state machine.

        The state machine's AT_GATE phase should enforce a 60-second
        turnaround delay before PUSHBACK is allowed.
        """
        from src.state_machine import TURNAROUND_DELAY_S

        assert TURNAROUND_DELAY_S == 60.0

        sm = FullLifecycleStateMachine(schema=schema, seed=42)
        state = sm.reset(task_id="integrated", episode_id="test-turnaround")

        aircraft = list(state.aircraft_states.values())[0]
        callsign = aircraft.callsign
        aircraft.assigned_gate = "GATE_A1"

        reached_at_gate = False
        at_gate_entry_step = 0
        max_steps = 1000

        for step in range(max_steps):
            if sm.is_terminal(state):
                break

            legal_actions = sm.get_legal_actions(state)
            if legal_actions:
                action = legal_actions[0]
            else:
                action = Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=callsign,
                )

            prev_phase = state.phase
            state, _ = sm.step(action)

            if (
                state.phase == LifecyclePhase.AT_GATE
                and prev_phase != LifecyclePhase.AT_GATE
            ):
                reached_at_gate = True
                at_gate_entry_step = step

            if reached_at_gate and state.phase != LifecyclePhase.AT_GATE:
                at_gate_duration = step - at_gate_entry_step
                assert at_gate_duration >= 60, (
                    f"AT_GATE should persist for at least 60 steps before transition, got {at_gate_duration}"
                )
                break

        assert reached_at_gate, "Never reached AT_GATE phase"


class TestIntegratedTaskDifficulty:
    """Tests to verify integrated task is marked as medium difficulty."""

    def test_integrated_task_difficulty_in_registry(self) -> None:
        """Test that integrated task is registered as medium difficulty."""
        from src.tasks.registry import TaskRegistry

        task = TaskRegistry.get("integrated")
        assert task.difficulty == "medium"

    def test_integrated_task_is_harder_than_departure(self) -> None:
        """Test that integrated task is harder than departure (easy)."""
        from src.tasks.registry import TaskRegistry

        departure = TaskRegistry.get("departure")
        integrated = TaskRegistry.get("integrated")

        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        assert (
            difficulty_order[integrated.difficulty]
            > difficulty_order[departure.difficulty]
        )

    def test_integrated_task_is_easier_than_arrival(self) -> None:
        """Test that integrated task is easier than arrival (hard)."""
        from src.tasks.registry import TaskRegistry

        arrival = TaskRegistry.get("arrival")
        integrated = TaskRegistry.get("integrated")

        difficulty_order = {"easy": 0, "medium": 1, "hard": 2}
        assert (
            difficulty_order[integrated.difficulty]
            < difficulty_order[arrival.difficulty]
        )
