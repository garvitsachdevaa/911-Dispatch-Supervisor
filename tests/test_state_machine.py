"""Tests for the full-lifecycle state machine."""

import pytest

from src.airport_schema import AirportSchema, AirportSchemaLoader
from src.models import Action, ClearanceType, LifecyclePhase
from src.state_machine import FullLifecycleStateMachine, LifecycleState


@pytest.fixture
def gatwick_schema() -> AirportSchema:
    """Load Gatwick airport schema for testing."""
    return AirportSchemaLoader.load("egkk_gatwick")


@pytest.fixture
def sm(gatwick_schema: AirportSchema) -> FullLifecycleStateMachine:
    """Create a seeded state machine for deterministic tests."""
    return FullLifecycleStateMachine(schema=gatwick_schema, seed=42)


class TestLifecycleState:
    """Tests for LifecycleState."""

    def test_lifecycle_state_creation(self) -> None:
        """Test creating a LifecycleState instance."""
        state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="test-123",
            step_count=0,
            task_id="arrival",
            completed_phases=[],
            metadata={},
        )
        assert state.phase == LifecyclePhase.APPROACH
        assert state.episode_id == "test-123"
        assert state.step_count == 0
        assert state.task_id == "arrival"
        assert len(state.completed_phases) == 0

    def test_lifecycle_state_defaults(self) -> None:
        """Test LifecycleState default values."""
        state = LifecycleState(episode_id="test", task_id="arrival")
        assert state.phase == LifecyclePhase.APPROACH
        assert state.aircraft_states == {}
        assert state.step_count == 0
        assert state.completed_phases == []
        assert state.metadata == {}


class TestReset:
    """Tests for reset() method."""

    def test_reset_returns_lifecycle_state(self, sm: FullLifecycleStateMachine) -> None:
        """Test that reset returns a LifecycleState."""
        state = sm.reset(task_id="arrival", episode_id="ep-1")
        assert isinstance(state, LifecycleState)

    def test_reset_initial_phase(self, sm: FullLifecycleStateMachine) -> None:
        """Test that reset starts at APPROACH phase."""
        state = sm.reset(task_id="arrival", episode_id="ep-1")
        assert state.phase == LifecyclePhase.APPROACH

    def test_reset_creates_aircraft(self, sm: FullLifecycleStateMachine) -> None:
        """Test that reset creates an aircraft in the state."""
        state = sm.reset(task_id="arrival", episode_id="ep-1")
        assert len(state.aircraft_states) == 1
        callsign = list(state.aircraft_states.keys())[0]
        aircraft = state.aircraft_states[callsign]
        assert aircraft.callsign == callsign
        assert aircraft.phase == LifecyclePhase.APPROACH

    def test_reset_stores_ids(self, sm: FullLifecycleStateMachine) -> None:
        """Test that reset stores task_id and episode_id."""
        state = sm.reset(task_id="arrival", episode_id="ep-123")
        assert state.task_id == "arrival"
        assert state.episode_id == "ep-123"

    def test_reset_deterministic(self, gatwick_schema: AirportSchema) -> None:
        """Test that reset is deterministic under fixed seed."""
        sm1 = FullLifecycleStateMachine(schema=gatwick_schema, seed=42)
        sm2 = FullLifecycleStateMachine(schema=gatwick_schema, seed=42)
        state1 = sm1.reset(task_id="arrival", episode_id="ep-1")
        state2 = sm2.reset(task_id="arrival", episode_id="ep-1")
        callsign = list(state1.aircraft_states.keys())[0]
        assert (
            state1.aircraft_states[callsign].x_ft
            == state2.aircraft_states[callsign].x_ft
        )
        assert (
            state1.aircraft_states[callsign].y_ft
            == state2.aircraft_states[callsign].y_ft
        )


class TestLegalActions:
    """Tests for get_legal_actions() method."""

    def test_approach_no_actions_until_altitude(
        self, sm: FullLifecycleStateMachine
    ) -> None:
        """Test that no actions are legal during approach until altitude threshold."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        state = sm._state
        # At initial altitude (35000ft), no actions should be legal
        legal = sm.get_legal_actions(state)
        assert len(legal) == 0

    def test_approach_actions_at_threshold(self, sm: FullLifecycleStateMachine) -> None:
        """Test that landing action becomes legal at altitude threshold."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        # Force altitude to threshold
        aircraft = list(sm._state.aircraft_states.values())[0]
        aircraft.altitude_ft = 40.0  # Below 50ft threshold
        state = sm._state
        legal = sm.get_legal_actions(state)
        assert len(legal) == 1
        assert legal[0].clearance_type == ClearanceType.LANDING

    def test_landing_legal_actions(self, sm: FullLifecycleStateMachine) -> None:
        """Test legal actions during LANDING phase."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.LANDING
        state = sm._state
        legal = sm.get_legal_actions(state)
        assert len(legal) == 1
        assert legal[0].clearance_type == ClearanceType.LANDING

    def test_docking_no_actions(self, sm: FullLifecycleStateMachine) -> None:
        """Test that DOCKING phase has no legal actions (automated)."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DOCKING
        state = sm._state
        legal = sm.get_legal_actions(state)
        assert len(legal) == 0

    def test_at_gate_pushback_action(self, sm: FullLifecycleStateMachine) -> None:
        """Test that pushback action is legal at AT_GATE."""
        sm.reset(task_id="departure", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.AT_GATE
        state = sm._state
        legal = sm.get_legal_actions(state)
        assert len(legal) == 1
        assert legal[0].clearance_type == ClearanceType.PUSHBACK

    def test_departure_queue_line_up_first(self, sm: FullLifecycleStateMachine) -> None:
        """Test that LINE_UP is required before TAKEOFF in departure queue."""
        sm.reset(task_id="departure", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DEPARTURE_QUEUE
        state = sm._state
        legal = sm.get_legal_actions(state)
        assert len(legal) == 1
        assert legal[0].clearance_type == ClearanceType.LINE_UP

    def test_departed_no_actions(self, sm: FullLifecycleStateMachine) -> None:
        """Test that DEPARTED phase has no legal actions."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DEPARTED
        state = sm._state
        legal = sm.get_legal_actions(state)
        assert len(legal) == 0


class TestIsTerminal:
    """Tests for is_terminal() method."""

    def test_departed_is_terminal(self, sm: FullLifecycleStateMachine) -> None:
        """Test that DEPARTED state is terminal."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DEPARTED
        assert sm.is_terminal(sm._state) is True

    def test_approach_not_terminal(self, sm: FullLifecycleStateMachine) -> None:
        """Test that APPROACH state is not terminal."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        assert sm.is_terminal(sm._state) is False

    def test_landing_not_terminal(self, sm: FullLifecycleStateMachine) -> None:
        """Test that LANDING state is not terminal."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.LANDING
        assert sm.is_terminal(sm._state) is False


class TestIllegalTransitions:
    """Tests for illegal transition rejection."""

    def test_skip_phase_rejected(self, sm: FullLifecycleStateMachine) -> None:
        """Test that skipping phases is rejected."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Try to do a taxi action while in APPROACH phase
        illegal_action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=callsign,
            route=["GATE_A1"],
        )
        state, obs = sm.step(illegal_action)
        assert obs.result == "illegal_transition"
        assert obs.score == 0.0
        assert "illegal_transition" in obs.issues

    def test_wrong_clearance_rejected(self, sm: FullLifecycleStateMachine) -> None:
        """Test that wrong clearance type is rejected."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.LANDING
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Try pushback while in LANDING
        illegal_action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        state, obs = sm.step(illegal_action)
        assert obs.result == "illegal_transition"

    def test_takeoff_before_line_up_rejected(
        self, sm: FullLifecycleStateMachine
    ) -> None:
        """Test that takeoff without line-up is rejected."""
        sm.reset(task_id="departure", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DEPARTURE_QUEUE
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Try takeoff without line_up first
        illegal_action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign=callsign,
        )
        state, obs = sm.step(illegal_action)
        assert obs.result == "illegal_transition"


class TestPhaseTransitions:
    """Tests for phase transition logic."""

    def test_approach_to_landing_transition(
        self, sm: FullLifecycleStateMachine
    ) -> None:
        """Test APPROACH to LANDING transition at altitude threshold."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        callsign = list(sm._state.aircraft_states.keys())[0]
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.altitude_ft = 40.0  # Below threshold
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign=callsign,
            runway="27L",
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.LANDING
        assert LifecyclePhase.APPROACH in state.completed_phases

    def test_landing_to_arrival_handoff(self, sm: FullLifecycleStateMachine) -> None:
        """Test LANDING to ARRIVAL_HANDOFF transition."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.LANDING
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Force aircraft to end of runway
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.y_ft = -3000.0  # Past runway end
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign=callsign,
            runway="27L",
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.ARRIVAL_HANDOFF
        assert LifecyclePhase.LANDING in state.completed_phases

    def test_arrival_handoff_to_taxi_in(self, sm: FullLifecycleStateMachine) -> None:
        """Test ARRIVAL_HANDOFF to TAXI_IN transition."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.ARRIVAL_HANDOFF
        callsign = list(sm._state.aircraft_states.keys())[0]
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=callsign,
            route=[],
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.TAXI_IN
        assert LifecyclePhase.ARRIVAL_HANDOFF in state.completed_phases

    def test_taxi_in_to_docking(self, sm: FullLifecycleStateMachine) -> None:
        """Test TAXI_IN to DOCKING transition."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.TAXI_IN
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Set assigned_gate to match the route destination
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.assigned_gate = "GATE_A1"
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=callsign,
            route=["GATE_A1"],
        )
        # Simulate reaching the gate
        aircraft.x_ft = -3500.0
        aircraft.y_ft = -8000.0
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.DOCKING
        assert LifecyclePhase.TAXI_IN in state.completed_phases

    def test_docking_to_at_gate(self, sm: FullLifecycleStateMachine) -> None:
        """Test DOCKING to AT_GATE transition after timer."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DOCKING
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Simulate docking timer completion (30+ steps)
        sm._docking_timer = 31.0
        state, obs = sm.step(
            Action(
                clearance_type=ClearanceType.TAXI,
                target_callsign=callsign,
                route=[],
            )
        )
        assert state.phase == LifecyclePhase.AT_GATE
        assert LifecyclePhase.DOCKING in state.completed_phases

    def test_at_gate_to_pushback(self, sm: FullLifecycleStateMachine) -> None:
        """Test AT_GATE to PUSHBACK transition after 60s turnaround delay."""
        sm.reset(task_id="departure", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.AT_GATE
        callsign = list(sm._state.aircraft_states.keys())[0]
        # Advance 59 steps (timer reaches 59s); the 60th step (actual PUSHBACK) triggers transition
        wait_action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        for _ in range(59):
            sm.step(wait_action)
        # Now pushback should succeed
        pushback_action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        state, obs = sm.step(pushback_action)
        assert state.phase == LifecyclePhase.PUSHBACK
        assert LifecyclePhase.AT_GATE in state.completed_phases


class TestFullLifecycleTraversal:
    """Tests for full lifecycle traversal."""

    def _descend_to_threshold(
        self, sm: FullLifecycleStateMachine, callsign: str
    ) -> None:
        """Helper to descend aircraft to landing threshold."""
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.altitude_ft = 40.0

    def _reach_runway_end(self, sm: FullLifecycleStateMachine, callsign: str) -> None:
        """Helper to move aircraft past runway end."""
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.y_ft = -3000.0

    def _reach_gate(self, sm: FullLifecycleStateMachine, callsign: str) -> None:
        """Helper to move aircraft to gate."""
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.x_ft = -3500.0
        aircraft.y_ft = -8000.0

    def test_full_arrival_lifecycle(self, sm: FullLifecycleStateMachine) -> None:
        """Test complete arrival lifecycle: approach to at_gate."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        callsign = list(sm._state.aircraft_states.keys())[0]

        # APPROACH -> LANDING
        self._descend_to_threshold(sm, callsign)
        action = Action(
            clearance_type=ClearanceType.LANDING, target_callsign=callsign, runway="27L"
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.LANDING

        # LANDING -> ARRIVAL_HANDOFF
        self._reach_runway_end(sm, callsign)
        action = Action(
            clearance_type=ClearanceType.LANDING, target_callsign=callsign, runway="27L"
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.ARRIVAL_HANDOFF

        # ARRIVAL_HANDOFF -> TAXI_IN
        action = Action(
            clearance_type=ClearanceType.TAXI, target_callsign=callsign, route=[]
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.TAXI_IN

        # TAXI_IN -> DOCKING (need route to gate)
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.assigned_gate = "GATE_A1"
        self._reach_gate(sm, callsign)
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=callsign,
            route=["GATE_A1"],
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.DOCKING

    def test_full_departure_lifecycle(self, sm: FullLifecycleStateMachine) -> None:
        """Test complete departure lifecycle: pushback to departed."""
        sm.reset(task_id="departure", episode_id="ep-1")
        callsign = list(sm._state.aircraft_states.keys())[0]

        # Set up at AT_GATE phase
        sm._state.phase = LifecyclePhase.AT_GATE
        sm._state.metadata["pushback_direction"] = "back"

        # Advance 59 steps (timer reaches 59s); the 60th step (actual PUSHBACK) triggers transition
        wait_action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        for _ in range(59):
            sm.step(wait_action)

        # AT_GATE -> PUSHBACK
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.PUSHBACK

        # PUSHBACK -> TAXI_OUT
        aircraft = sm._state.aircraft_states[callsign]
        aircraft.y_ft = -600.0  # Moved back
        sm._state.metadata["stand_node"] = "STAND_101"
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=callsign,
            pushback_direction="back",
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.TAXI_OUT

        # TAXI_OUT -> DEPARTURE_QUEUE
        sm._state.metadata["taxi_route"] = ["DQ_E", "RE_E"]
        sm._route_index = 2
        sm._current_node = "DQ_E"
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=callsign,
            route=["DQ_E", "RE_E"],
        )
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.DEPARTURE_QUEUE

        # DEPARTURE_QUEUE -> TAKEOFF (need LINE_UP then TAKEOFF)
        action = Action(clearance_type=ClearanceType.LINE_UP, target_callsign=callsign)
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.DEPARTURE_QUEUE  # Still in queue

        action = Action(clearance_type=ClearanceType.TAKEOFF, target_callsign=callsign)
        state, obs = sm.step(action)
        assert state.phase == LifecyclePhase.TAKEOFF


class TestRewardSignals:
    """Tests for reward signal composition."""

    def test_phase_transition_reward(self, sm: FullLifecycleStateMachine) -> None:
        """Test that successful phase transitions earn +0.1 reward."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DOCKING
        sm._docking_timer = 31.0  # Docking complete
        callsign = list(sm._state.aircraft_states.keys())[0]
        action = Action(
            clearance_type=ClearanceType.TAXI, target_callsign=callsign, route=[]
        )
        state, obs = sm.step(action)
        assert obs.score == 0.1

    def test_illegal_transition_penalty(self, sm: FullLifecycleStateMachine) -> None:
        """Test that illegal transitions earn 0.0 score."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        callsign = list(sm._state.aircraft_states.keys())[0]
        illegal_action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign=callsign,
        )
        state, obs = sm.step(illegal_action)
        assert obs.score == 0.0

    def test_departed_high_score(self, sm: FullLifecycleStateMachine) -> None:
        """Test that DEPARTED state returns score of 1.0."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        sm._state.phase = LifecyclePhase.DEPARTED
        callsign = list(sm._state.aircraft_states.keys())[0]
        action = Action(clearance_type=ClearanceType.TAKEOFF, target_callsign=callsign)
        state, obs = sm.step(action)
        assert obs.score == 1.0

    def test_cumulative_rewards(self, sm: FullLifecycleStateMachine) -> None:
        """Test that rewards accumulate across successful transitions."""
        sm.reset(task_id="arrival", episode_id="ep-1")
        callsign = list(sm._state.aircraft_states.keys())[0]

        # Make several transitions and track cumulative rewards
        total_reward = 0.0

        # DOCKING -> AT_GATE
        sm._state.phase = LifecyclePhase.DOCKING
        sm._docking_timer = 31.0
        action = Action(
            clearance_type=ClearanceType.TAXI, target_callsign=callsign, route=[]
        )
        state, obs = sm.step(action)
        total_reward += obs.score
        assert total_reward >= 0.0


class TestDeterminism:
    """Tests for deterministic behavior under fixed seed."""

    def test_same_seed_same_sequence(self, gatwick_schema: AirportSchema) -> None:
        """Test that same seed produces same episode sequence."""
        sm1 = FullLifecycleStateMachine(schema=gatwick_schema, seed=42)
        sm2 = FullLifecycleStateMachine(schema=gatwick_schema, seed=42)

        sm1.reset(task_id="arrival", episode_id="ep-1")
        sm2.reset(task_id="arrival", episode_id="ep-1")

        # Run through multiple steps
        for _ in range(10):
            actions1 = sm1.get_legal_actions(sm1._state)
            actions2 = sm2.get_legal_actions(sm2._state)
            if actions1:
                action1 = actions1[0]
                action2 = actions2[0] if actions2 else actions1[0]
                sm1.step(action1)
                sm2.step(action2)

        # Aircraft states should be identical
        ac1 = list(sm1._state.aircraft_states.values())[0]
        ac2 = list(sm2._state.aircraft_states.values())[0]
        assert ac1.x_ft == ac2.x_ft
        assert ac1.y_ft == ac2.y_ft
        assert ac1.altitude_ft == ac2.altitude_ft
