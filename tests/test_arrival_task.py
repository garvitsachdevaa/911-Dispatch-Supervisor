"""Tests for ArrivalTask and ArrivalGrader."""

import pytest

from src.airport_schema import (
    AirportNode,
    AirportSchema,
    AirportEdge,
    EdgeMovement,
    NodeType,
)
from src.models import Action, ClearanceType, LifecyclePhase, Observation
from src.rewards import RewardCalculator
from src.state_machine import FullLifecycleStateMachine, LifecycleState
from src.tasks.arrival import ArrivalGrader, ArrivalTask


def _make_gatwick_schema() -> AirportSchema:
    return AirportSchema(
        airport_code="EGKK",
        nodes={
            "APP_FIX_27L": AirportNode(
                id="APP_FIX_27L", node_type=NodeType.APPROACH_FIX, x_ft=0.0, y_ft=5000.0
            ),
            "THR_27L": AirportNode(
                id="THR_27L", node_type=NodeType.LANDING_THRESHOLD, x_ft=0.0, y_ft=0.0
            ),
            "RWY_EXIT_1": AirportNode(
                id="RWY_EXIT_1",
                node_type=NodeType.EXIT_TAXIWAY,
                x_ft=100.0,
                y_ft=-500.0,
            ),
            "TWY_C": AirportNode(
                id="TWY_C", node_type=NodeType.TAXI_POINT, x_ft=200.0, y_ft=-1000.0
            ),
            "TWY_D": AirportNode(
                id="TWY_D", node_type=NodeType.TAXI_POINT, x_ft=300.0, y_ft=-2000.0
            ),
            "GATE_B2": AirportNode(
                id="GATE_B2", node_type=NodeType.GATE, x_ft=400.0, y_ft=-3000.0
            ),
            "stand_1": AirportNode(
                id="stand_1", node_type=NodeType.STAND, x_ft=400.0, y_ft=-3100.0
            ),
        },
        edges=[
            AirportEdge(
                from_node="APP_FIX_27L",
                to_node="THR_27L",
                movement_type=EdgeMovement.APPROACH,
                distance_ft=5000.0,
                max_speed_kt=250.0,
            ),
            AirportEdge(
                from_node="THR_27L",
                to_node="RWY_EXIT_1",
                movement_type=EdgeMovement.LANDING,
                distance_ft=500.0,
                max_speed_kt=160.0,
            ),
            AirportEdge(
                from_node="RWY_EXIT_1",
                to_node="TWY_C",
                movement_type=EdgeMovement.EXIT_RUNWAY,
                distance_ft=500.0,
                max_speed_kt=20.0,
            ),
            AirportEdge(
                from_node="TWY_C",
                to_node="TWY_D",
                movement_type=EdgeMovement.TAXI,
                distance_ft=1000.0,
                max_speed_kt=20.0,
            ),
            AirportEdge(
                from_node="TWY_D",
                to_node="GATE_B2",
                movement_type=EdgeMovement.TAXI,
                distance_ft=1000.0,
                max_speed_kt=20.0,
            ),
        ],
        runways=[{"id": "27L", "heading_deg": 270.0}],
        gates=[{"id": "GATE_B2", "x_ft": 400.0, "y_ft": -3000.0}],
    )


class TestArrivalTask:
    def test_arrival_task_initializes(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        assert task is not None

    def test_arrival_task_has_state_machine(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        assert hasattr(task, "state_machine")
        assert isinstance(task.state_machine, FullLifecycleStateMachine)

    def test_arrival_task_starts_at_approach_phase(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        state = task.reset(episode_id="test-001")
        assert state.phase == LifecyclePhase.APPROACH

    def test_arrival_task_aircraft_starts_at_high_altitude(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        state = task.reset(episode_id="test-001")
        aircraft = next(iter(state.aircraft_states.values()))
        assert aircraft.altitude_ft > 0.0
        assert aircraft.phase == LifecyclePhase.APPROACH

    def test_arrival_task_task_id_is_arrival(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        state = task.reset(episode_id="test-001")
        assert state.task_id == "arrival"

    def test_arrival_task_completes_at_at_gate(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        task.reset(episode_id="test-001")

        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            state = task.state_machine._state
            callsign = list(state.aircraft_states.keys())[0]
            legal_actions = task.state_machine.get_legal_actions(state)

            if legal_actions:
                action = legal_actions[0]
            elif state.phase == LifecyclePhase.DOCKING:
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign=callsign,
                    route=[],
                )
            else:
                action = Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=callsign,
                    runway="27L",
                )

            state, obs = task.step(action)
            done = task.is_terminal(state)
            steps += 1

        assert state.phase == LifecyclePhase.AT_GATE
        assert steps < max_steps

    def test_arrival_task_is_deterministic(self) -> None:
        schema = _make_gatwick_schema()
        task1 = ArrivalTask(airport_schema=schema, seed=12345)
        task2 = ArrivalTask(airport_schema=schema, seed=12345)

        state1 = task1.reset(episode_id="test-001")
        state2 = task2.reset(episode_id="test-001")

        assert state1.phase == state2.phase
        assert state1.episode_id == state2.episode_id

        task1.reset(episode_id="test-001")
        task2.reset(episode_id="test-001")

        for _ in range(10):
            state = task1.state_machine._state
            legal_actions = task1.state_machine.get_legal_actions(state)
            if legal_actions:
                action = legal_actions[0]
                task1.step(action)
                task2.step(action)

        aircraft1 = next(iter(task1.state_machine._state.aircraft_states.values()))
        aircraft2 = next(iter(task2.state_machine._state.aircraft_states.values()))
        assert aircraft1.altitude_ft == aircraft2.altitude_ft


class TestArrivalGrader:
    def test_arrival_grader_exists(self) -> None:
        grader = ArrivalGrader()
        assert grader is not None

    def test_grade_returns_float(self) -> None:
        grader = ArrivalGrader()
        rewards = [0.5, 0.6, 0.7]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        assert isinstance(result, float)

    def test_grade_in_range(self) -> None:
        grader = ArrivalGrader()
        rewards = [0.5, 0.6, 0.7]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        assert 0.0 <= result <= 1.0

    def test_grade_with_all_zeros(self) -> None:
        grader = ArrivalGrader()
        rewards = [0.0, 0.0, 0.0]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        assert abs(result - 0.1) < 1e-9

    def test_grade_with_all_ones(self) -> None:
        grader = ArrivalGrader()
        rewards = [1.0, 1.0, 1.0]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        assert result == 1.0

    def test_completion_bonus_applied(self) -> None:
        grader = ArrivalGrader()
        rewards = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        expected = 0.5 + 0.1
        assert abs(result - expected) < 1e-9

    def test_completion_bonus_not_applied_partial(self) -> None:
        grader = ArrivalGrader()
        rewards = [0.5, 0.5, 0.5, 0.5, 0.5]
        state = LifecycleState(
            phase=LifecyclePhase.DOCKING,
            aircraft_states={},
            episode_id="test",
            step_count=80,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        assert abs(result - 0.5) < 1e-9

    def test_score_clamped_to_one(self) -> None:
        grader = ArrivalGrader()
        rewards = [1.0, 1.0, 1.0]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result = grader.grade(state, rewards)
        assert result == 1.0

    def test_deterministic_grading(self) -> None:
        grader = ArrivalGrader()
        rewards = [0.5, 0.6, 0.7, 0.8]
        state = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        result1 = grader.grade(state, rewards)
        result2 = grader.grade(state, rewards)
        assert result1 == result2


class TestArrivalIntegration:
    def test_safe_arrival_full_episode(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        grader = ArrivalGrader()

        state = task.reset(episode_id="safe-arrival-001")
        episode_rewards = []
        done = False
        steps = 0
        max_steps = 1000

        while not done and steps < max_steps:
            current_state = task.state_machine._state
            callsign = list(current_state.aircraft_states.keys())[0]
            legal_actions = task.state_machine.get_legal_actions(current_state)

            if legal_actions:
                action = legal_actions[0]
            elif current_state.phase == LifecyclePhase.DOCKING:
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign=callsign,
                    route=[],
                )
            else:
                action = Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=callsign,
                    runway="27L",
                )

            state, obs = task.step(action)

            calc = RewardCalculator()
            aircraft = next(iter(state.aircraft_states.values()), None)
            if aircraft:
                dummy_action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign=aircraft.callsign,
                    route=[],
                )
                _, step_reward = calc.compute_reward(state, dummy_action, obs)
                episode_rewards.append(step_reward)

            done = task.is_terminal(state)
            steps += 1

        final_score = grader.grade(state, episode_rewards)

        assert state.phase == LifecyclePhase.AT_GATE
        assert len(episode_rewards) > 0
        assert 0.0 <= final_score <= 1.0

    def test_unsafe_arrival_receives_low_score(self) -> None:
        schema = _make_gatwick_schema()
        task = ArrivalTask(airport_schema=schema, seed=42)
        grader = ArrivalGrader()

        state = task.reset(episode_id="unsafe-arrival-001")
        callsign = list(state.aircraft_states.keys())[0]

        calc = RewardCalculator()
        episode_rewards = []

        for _ in range(5):
            state, obs = task.step(
                Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=callsign,
                    runway="27L",
                )
            )
            _, step_reward = calc.compute_reward(
                state,
                Action(
                    clearance_type=ClearanceType.LANDING,
                    target_callsign=callsign,
                    runway="27L",
                ),
                obs,
            )
            episode_rewards.append(step_reward)

        unsafe_obs = Observation(
            result="runway_incursion",
            score=0.0,
            phraseology_ok=False,
            issues=["runway_incursion"],
        )
        state, _ = task.step(
            Action(
                clearance_type=ClearanceType.TAXI,
                target_callsign=callsign,
                route=["RWY_EXIT_1"],
            )
        )

        _, unsafe_reward = calc.compute_reward(
            state,
            Action(
                clearance_type=ClearanceType.TAXI,
                target_callsign=callsign,
                route=["RWY_EXIT_1"],
            ),
            unsafe_obs,
        )
        episode_rewards.append(unsafe_reward)

        final_score = grader.grade(state, episode_rewards)
        assert final_score < 0.8

    def test_score_range_always_valid(self) -> None:
        grader = ArrivalGrader()
        test_cases = [
            [],
            [0.0],
            [1.0],
            [0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.4, 0.6, 0.8],
            [0.9, 0.9, 0.9],
        ]
        state_at_gate = LifecycleState(
            phase=LifecyclePhase.AT_GATE,
            aircraft_states={},
            episode_id="test",
            step_count=100,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
                LifecyclePhase.AT_GATE,
            ],
            metadata={},
        )
        for rewards in test_cases:
            result = grader.grade(state_at_gate, rewards)
            assert 0.0 <= result <= 1.0, (
                f"Score {result} out of range for rewards {rewards}"
            )
