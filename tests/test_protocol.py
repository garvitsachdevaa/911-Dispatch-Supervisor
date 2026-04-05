"""Tests for dispatch protocol validation."""

from __future__ import annotations

from src.city_schema import CitySchema
from src.models import (
    Action,
    DispatchAction,
    IncidentSeverity,
    IncidentState,
    IncidentStatus,
    IncidentType,
    State,
    UnitState,
    UnitStatus,
    UnitType,
)
from src.protocol import DispatchProtocolValidator


def _schema() -> CitySchema:
    return CitySchema(
        city_name="Testopolis",
        grid_size=[100, 100],
        districts=["a", "b"],
        units=[],
        unit_speeds={UnitType.MEDIC: 1.0, UnitType.ENGINE: 1.0},
        default_required_units={IncidentType.CARDIAC_ARREST: [UnitType.MEDIC]},
    )


def _state() -> State:
    unit = UnitState(
        unit_id="ENG-1",
        unit_type=UnitType.ENGINE,
        status=UnitStatus.AVAILABLE,
        location_x=0.0,
        location_y=0.0,
        assigned_incident_id=None,
        eta_seconds=0.0,
        crew_count=4,
    )
    inc = IncidentState(
        incident_id="INC-001",
        incident_type=IncidentType.CARDIAC_ARREST,
        severity=IncidentSeverity.PRIORITY_2,
        location_x=10.0,
        location_y=10.0,
        reported_at_step=0,
        units_assigned=[],
        status=IncidentStatus.PENDING,
        survival_clock=100.0,
    )
    return State(
        units={unit.unit_id: unit},
        incidents={inc.incident_id: inc},
        episode_id="ep",
        step_count=0,
        task_id="single_incident",
        city_time=0.0,
        metadata={},
    )


def test_type_mismatch_is_warning_not_error() -> None:
    validator = DispatchProtocolValidator()
    result = validator.validate(
        _schema(),
        _state(),
        Action(action_type=DispatchAction.DISPATCH, unit_id="ENG-1", incident_id="INC-001"),
    )
    assert result.ok is True
    assert any(i.startswith("warn:") for i in result.issues)


def test_upgrade_requires_higher_override() -> None:
    validator = DispatchProtocolValidator()
    st = _state()
    st.incidents["INC-001"].severity = IncidentSeverity.PRIORITY_2
    bad = validator.validate(
        _schema(),
        st,
        Action(
            action_type=DispatchAction.UPGRADE,
            unit_id="ENG-1",
            incident_id="INC-001",
            priority_override=IncidentSeverity.PRIORITY_3,
        ),
    )
    assert bad.ok is False


def test_mutual_aid_disallowed_when_local_available() -> None:
    validator = DispatchProtocolValidator()
    st = _state()
    result = validator.validate(
        _schema(),
        st,
        Action(action_type=DispatchAction.MUTUAL_AID, unit_id="ENG-1", incident_id="INC-001"),
    )
    assert result.ok is False
