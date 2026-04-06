"""Tests for reward engine and grader primitives (dispatch domain)."""

from __future__ import annotations

import pytest

from src.models import (
    Action,
    DispatchAction,
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
    IncidentType,
    Observation,
    State,
    UnitState,
    UnitStatus,
    UnitType,
)
from src.rewards import RewardCalculator, RewardSignal


def _state_with_one_dispatch() -> State:
    unit = UnitState(
        unit_id="MED-1",
        unit_type=UnitType.MEDIC,
        status=UnitStatus.DISPATCHED,
        location_x=0.0,
        location_y=0.0,
        assigned_incident_id="INC-001",
        eta_seconds=200.0,
        crew_count=2,
    )
    inc = IncidentState(
        incident_id="INC-001",
        incident_type=IncidentType.CARDIAC_ARREST,
        severity=IncidentSeverity.PRIORITY_1,
        location_x=10.0,
        location_y=10.0,
        reported_at_step=0,
        units_assigned=["MED-1"],
        status=IncidentStatus.RESPONDING,
        survival_clock=100.0,
    )
    return State(
        units={"MED-1": unit},
        incidents={"INC-001": inc},
        episode_id="ep",
        step_count=1,
        task_id="single_incident",
        city_time=30.0,
        metadata={
            "default_required_units": {"IncidentType.CARDIAC_ARREST": ["UnitType.MEDIC"]},
            "districts": ["a", "b"],
            "grid_size": [100, 100],
        },
    )


def test_reward_signal_requires_fields() -> None:
    with pytest.raises(Exception):
        RewardSignal()  # type: ignore[call-arg]


def test_compute_reward_returns_tuple() -> None:
    calc = RewardCalculator()
    state = _state_with_one_dispatch()
    action = Action(
        action_type=DispatchAction.DISPATCH,
        unit_id="MED-1",
        incident_id="INC-001",
        notes="DISPATCH MED-1 -> INC-001",
    )
    obs = Observation(result="ok", score=0.8, protocol_ok=True, issues=[])
    signal, total = calc.compute_reward(state, action, obs)
    assert isinstance(signal, RewardSignal)
    # Fixture metadata stores enum-ish strings (e.g. "IncidentType.CARDIAC_ARREST").
    # Triage should still award full credit for a correct match.
    assert signal.triage == 1.0
    assert signal.protocol == 1.0
    assert 0.0 <= total <= 1.0


def test_protocol_reward_uses_phraseology_notes() -> None:
    calc = RewardCalculator()
    state = _state_with_one_dispatch()
    obs = Observation(result="ok", score=0.8, protocol_ok=True, issues=[])

    # No notes => neutral.
    action_no_notes = Action(action_type=DispatchAction.DISPATCH, unit_id="MED-1", incident_id="INC-001")
    signal, _ = calc.compute_reward(state, action_no_notes, obs)
    assert signal.protocol == 0.5

    # Wrong notes => poor phraseology score.
    action_bad_notes = Action(
        action_type=DispatchAction.DISPATCH,
        unit_id="MED-1",
        incident_id="INC-001",
        notes="send car 12",
    )
    signal2, _ = calc.compute_reward(state, action_bad_notes, obs)
    assert signal2.protocol == 0.0


def test_weights_sum_to_one() -> None:
    calc = RewardCalculator()
    assert abs(sum(calc.weights.values()) - 1.0) < 1e-9
