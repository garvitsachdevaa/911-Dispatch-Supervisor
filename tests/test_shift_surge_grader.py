"""Unit tests for shift_surge episode grading."""

from __future__ import annotations

from src.models import (
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
    IncidentType,
    State,
    UnitState,
    UnitStatus,
    UnitType,
)
from src.tasks.shift_surge import ShiftSurgeGrader


def _base_state() -> State:
    units = {
        "MED-1": UnitState(
            unit_id="MED-1",
            unit_type=UnitType.MEDIC,
            status=UnitStatus.AVAILABLE,
            location_x=10.0,
            location_y=10.0,
            assigned_incident_id=None,
            eta_seconds=0.0,
            crew_count=2,
        ),
        "ENG-1": UnitState(
            unit_id="ENG-1",
            unit_type=UnitType.ENGINE,
            status=UnitStatus.AVAILABLE,
            location_x=50.0,
            location_y=50.0,
            assigned_incident_id=None,
            eta_seconds=0.0,
            crew_count=4,
        ),
        "PAT-1": UnitState(
            unit_id="PAT-1",
            unit_type=UnitType.PATROL,
            status=UnitStatus.AVAILABLE,
            location_x=90.0,
            location_y=10.0,
            assigned_incident_id=None,
            eta_seconds=0.0,
            crew_count=2,
        ),
    }

    incidents = {
        "INC-001": IncidentState(
            incident_id="INC-001",
            incident_type=IncidentType.CARDIAC_ARREST,
            severity=IncidentSeverity.PRIORITY_1,
            location_x=12.0,
            location_y=12.0,
            reported_at_step=0,
            units_assigned=[],
            status=IncidentStatus.PENDING,
            survival_clock=600.0,
        ),
        "INC-002": IncidentState(
            incident_id="INC-002",
            incident_type=IncidentType.STRUCTURE_FIRE,
            severity=IncidentSeverity.PRIORITY_2,
            location_x=55.0,
            location_y=48.0,
            reported_at_step=0,
            units_assigned=[],
            status=IncidentStatus.PENDING,
            survival_clock=1200.0,
        ),
    }

    return State(
        units=units,
        incidents=incidents,
        episode_id="ep",
        step_count=10,
        task_id="shift_surge",
        city_time=300.0,
        metadata={
            "districts": ["a", "b", "c"],
            "grid_size": [100, 100],
            "p1_seen": ["INC-001"],
            "resolved_incidents": [],
            "failed_incidents": [],
        },
    )


def test_shift_surge_grader_rewards_good_outcome() -> None:
    state = _base_state()
    state.incidents["INC-001"].status = IncidentStatus.RESOLVED
    state.incidents["INC-002"].status = IncidentStatus.RESOLVED
    state.metadata["resolved_incidents"] = ["INC-001", "INC-002"]

    score = ShiftSurgeGrader().grade(state, rewards=[0.9] * 10)
    assert 0.8 <= score <= 1.0


def test_shift_surge_grader_penalizes_failures_and_backlog() -> None:
    state = _base_state()
    state.incidents["INC-001"].status = IncidentStatus.ESCALATED
    state.incidents["INC-002"].status = IncidentStatus.RESPONDING
    state.metadata["failed_incidents"] = ["INC-001"]

    score = ShiftSurgeGrader().grade(state, rewards=[0.2] * 10)
    assert 0.0 <= score <= 0.4
