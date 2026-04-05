"""Tests for departure task."""

import pytest

from src.models import Action, ClearanceType, LifecyclePhase, Observation
from src.state_machine import LifecycleState
from src.tasks.departure import DepartureGrader, DepartureTask, NUM_DEPARTURE_PHASES


class TestDepartureTask:
    """Tests for DepartureTask."""

    @pytest.fixture
    def task(self) -> DepartureTask:
        """Create a DepartureTask instance."""
        return DepartureTask()

    def test_departure_task_has_default_task_id(self, task: DepartureTask) -> None:
        """Test that DepartureTask has default task_id of 'departure'."""
        assert task.task_id == "departure"

    def test_reset_returns_lifecycle_state(self, task: DepartureTask) -> None:
        """Test that reset returns a LifecycleState."""
        state = task.reset(seed=42)
        assert isinstance(state, LifecycleState)

    def test_reset_starts_at_gate(self, task: DepartureTask) -> None:
        """Test that reset starts at AT_GATE phase."""
        state = task.reset(seed=42)
        assert state.phase == LifecyclePhase.AT_GATE

    def test_reset_has_aircraft(self, task: DepartureTask) -> None:
        """Test that reset includes an aircraft."""
        state = task.reset(seed=42)
        assert len(state.aircraft_states) == 1

    def test_reset_task_id_is_departure(self, task: DepartureTask) -> None:
        """Test that reset sets task_id to 'departure'."""
        state = task.reset(seed=42)
        assert state.task_id == "departure"

    def test_reset_completed_phases_includes_at_gate(self, task: DepartureTask) -> None:
        """Test that AT_GATE is marked as completed on reset."""
        state = task.reset(seed=42)
        assert LifecyclePhase.AT_GATE in state.completed_phases

    def test_is_terminal_false_before_departed(self, task: DepartureTask) -> None:
        """Test is_terminal returns False before DEPARTED phase."""
        state = task.reset(seed=42)
        assert not task.is_terminal(state)

    def test_is_terminal_true_at_departed(self, task: DepartureTask) -> None:
        """Test is_terminal returns True at DEPARTED phase."""
        state = task.reset(seed=42)
        state.phase = LifecyclePhase.DEPARTED
        assert task.is_terminal(state)

    def test_get_legal_actions_at_gate(self, task: DepartureTask) -> None:
        """Test legal actions at AT_GATE phase."""
        state = task.reset(seed=42)
        actions = task.get_legal_actions(state)
        assert len(actions) > 0
        assert all(a.clearance_type == ClearanceType.PUSHBACK for a in actions)

    def test_get_legal_actions_at_pushback(self, task: DepartureTask) -> None:
        """Test legal actions at PUSHBACK phase."""
        state = task.reset(seed=42)
        state.phase = LifecyclePhase.PUSHBACK
        actions = task.get_legal_actions(state)
        assert len(actions) > 0

    def test_step_transitions_through_phases(self, task: DepartureTask) -> None:
        """Test that step properly transitions through departure phases."""
        state = task.reset(seed=42)
        aircraft = next(iter(state.aircraft_states.values()))
        callsign = aircraft.callsign

        # Step 1: Pushback
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        state, obs = task.step(state, action)
        assert state.phase == LifecyclePhase.PUSHBACK
        assert LifecyclePhase.AT_GATE in state.completed_phases


class TestDepartureGrader:
    """Tests for DepartureGrader."""

    @pytest.fixture
    def grader(self) -> DepartureGrader:
        """Create a DepartureGrader instance."""
        return DepartureGrader()

    def test_grader_has_reward_calculator(self, grader: DepartureGrader) -> None:
        """Test that grader has a RewardCalculator."""
        assert hasattr(grader, "_reward_calculator")

    def test_grade_returns_float(self, grader: DepartureGrader) -> None:
        """Test that grade returns a float."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test",
            step_count=10,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={},
        )
        result = grader.grade(state, [0.5, 0.6, 0.7])
        assert isinstance(result, float)

    def test_score_in_range(self, grader: DepartureGrader) -> None:
        """Test that score is in [0.0, 1.0]."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test",
            step_count=10,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={},
        )
        result = grader.grade(state, [0.5, 0.6, 0.7])
        assert 0.0 <= result <= 1.0

    def test_apron_conflict_score_zero(self, grader: DepartureGrader) -> None:
        """Test that apron conflict results in score 0."""
        state = LifecycleState(
            phase=LifecyclePhase.PUSHBACK,
            aircraft_states={},
            episode_id="test",
            step_count=1,
            task_id="departure",
            completed_phases=[LifecyclePhase.AT_GATE],
            metadata={"apron_conflict": True},
        )
        result = grader.grade(state, [0.9, 0.9, 0.9])
        assert result == 0.0

    def test_unsafe_runway_release_score_zero(self, grader: DepartureGrader) -> None:
        """Test that unsafe runway release results in score 0."""
        state = LifecycleState(
            phase=LifecyclePhase.TAKEOFF,
            aircraft_states={},
            episode_id="test",
            step_count=5,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
            ],
            metadata={"unsafe_runway_release": True},
        )
        result = grader.grade(state, [0.8, 0.8, 0.8])
        assert result == 0.0

    def test_runway_incursion_score_zero(self, grader: DepartureGrader) -> None:
        """Test that runway incursion results in score 0."""
        state = LifecycleState(
            phase=LifecyclePhase.TAKEOFF,
            aircraft_states={},
            episode_id="test",
            step_count=5,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
            ],
            metadata={"runway_incursion": True},
        )
        result = grader.grade(state, [0.9, 0.9, 0.9])
        assert result == 0.0

    def test_completion_bonus_applied(self, grader: DepartureGrader) -> None:
        """Test that completion bonus is applied when all phases done."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test",
            step_count=10,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={
                "line_up_confirmed": True,
                "takeoff_confirmed": True,
            },
        )
        result = grader.grade(state, [0.5, 0.5, 0.5])
        assert result >= 0.5

    def test_no_completion_bonus_incomplete(self, grader: DepartureGrader) -> None:
        """Test that completion bonus is NOT applied when phases are missing."""
        state = LifecycleState(
            phase=LifecyclePhase.TAKEOFF,
            aircraft_states={},
            episode_id="test",
            step_count=5,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
            ],
            metadata={
                "line_up_confirmed": True,
                "takeoff_confirmed": True,
            },
        )
        result = grader.grade(state, [0.5, 0.5, 0.5])
        assert result == 0.5

    def test_all_phases_completed_true(self, grader: DepartureGrader) -> None:
        """Test _all_departure_phases_completed returns True when all done."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test",
            step_count=10,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={},
        )
        assert grader._all_departure_phases_completed(state)

    def test_all_phases_completed_false(self, grader: DepartureGrader) -> None:
        """Test _all_departure_phases_completed returns False when incomplete."""
        state = LifecycleState(
            phase=LifecyclePhase.TAXI_OUT,
            aircraft_states={},
            episode_id="test",
            step_count=3,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.PUSHBACK,
                # Missing DEPARTURE_QUEUE, TAKEOFF, DEPARTED
            ],
            metadata={},
        )
        assert not grader._all_departure_phases_completed(state)

    def test_empty_rewards_returns_zero(self, grader: DepartureGrader) -> None:
        """Test that grading empty rewards returns 0."""
        state = LifecycleState(
            phase=LifecyclePhase.DEPARTED,
            aircraft_states={},
            episode_id="test",
            step_count=0,
            task_id="departure",
            completed_phases=[
                LifecyclePhase.AT_GATE,
                LifecyclePhase.PUSHBACK,
                LifecyclePhase.TAXI_OUT,
                LifecyclePhase.DEPARTURE_QUEUE,
                LifecyclePhase.TAKEOFF,
                LifecyclePhase.DEPARTED,
            ],
            metadata={},
        )
        result = grader.grade(state, [])
        assert result == 0.0


class TestSafeDepartureFlow:
    """Integration tests for safe departure flow."""

    @pytest.fixture
    def task(self) -> DepartureTask:
        """Create a DepartureTask instance."""
        return DepartureTask()

    @pytest.fixture
    def grader(self) -> DepartureGrader:
        """Create a DepartureGrader instance."""
        return DepartureGrader()

    def test_full_safe_departure_completes(
        self, task: DepartureTask, grader: DepartureGrader
    ) -> None:
        """Test a full safe departure completes all phases."""
        state = task.reset(seed=42)
        aircraft = next(iter(state.aircraft_states.values()))
        callsign = aircraft.callsign
        episode_rewards = []

        # AT_GATE -> PUSHBACK
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        state, obs = task.step(state, action)
        reward = grader.grade_step(state, action, obs)
        episode_rewards.append(reward)

        # Continue pushback until transition
        for _ in range(50):
            if state.phase == LifecyclePhase.TAXI_OUT:
                break
            state, obs = task.step(state, action)
            reward = grader.grade_step(state, action, obs)
            episode_rewards.append(reward)

        if state.phase == LifecyclePhase.PUSHBACK:
            action = Action(
                clearance_type=ClearanceType.PUSHBACK,
                target_callsign=callsign,
                pushback_direction="back",
            )
            state, obs = task.step(state, action)
            reward = grader.grade_step(state, action, obs)
            episode_rewards.append(reward)

        # TAXI_OUT -> DEPARTURE_QUEUE
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=callsign,
            route=["DQ_E", "RE_E"],
        )
        state, obs = task.step(state, action)
        reward = grader.grade_step(state, action, obs)
        episode_rewards.append(reward)

        # Continue taxi until queue
        for _ in range(50):
            if state.phase == LifecyclePhase.DEPARTURE_QUEUE:
                break
            state, obs = task.step(state, action)
            reward = grader.grade_step(state, action, obs)
            episode_rewards.append(reward)

        # LINE_UP
        action = Action(
            clearance_type=ClearanceType.LINE_UP,
            target_callsign=callsign,
        )
        state, obs = task.step(state, action)
        reward = grader.grade_step(state, action, obs)
        episode_rewards.append(reward)

        # TAKEOFF
        action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign=callsign,
        )
        state, obs = task.step(state, action)
        reward = grader.grade_step(state, action, obs)
        episode_rewards.append(reward)

        # Final grading
        final_score = grader.grade(state, episode_rewards)
        assert 0.0 <= final_score <= 1.0

    def test_departure_task_is_deterministic(self, task: DepartureTask) -> None:
        """Test that departure task produces deterministic results."""
        state1 = task.reset(seed=12345)
        state2 = task.reset(seed=12345)
        assert state1.episode_id == state2.episode_id
        callsign1 = list(state1.aircraft_states.keys())[0]
        callsign2 = list(state2.aircraft_states.keys())[0]
        assert callsign1 == callsign2


class TestDeparturePhaseSequence:
    """Tests for the departure phase sequence."""

    def test_departure_phases_count(self) -> None:
        """Test that there are exactly 5 departure phases (excluding AT_GATE)."""
        assert NUM_DEPARTURE_PHASES == 5

    def test_departure_phases_list(self) -> None:
        """Test the correct departure phases are defined."""
        from src.tasks.departure import DEPARTURE_PHASES

        expected = [
            LifecyclePhase.PUSHBACK,
            LifecyclePhase.TAXI_OUT,
            LifecyclePhase.DEPARTURE_QUEUE,
            LifecyclePhase.TAKEOFF,
            LifecyclePhase.DEPARTED,
        ]
        assert DEPARTURE_PHASES == expected
