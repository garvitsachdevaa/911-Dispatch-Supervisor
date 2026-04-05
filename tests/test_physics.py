"""Tests for city-grid physics utilities."""

from __future__ import annotations

import pytest

from src.models import IncidentSeverity, IncidentState, IncidentStatus, IncidentType, UnitState, UnitStatus, UnitType
from src.physics import check_arrival, compute_coverage_score, compute_eta, move_unit_toward


def _unit(x: float, y: float) -> UnitState:
    return UnitState(
        unit_id="MED-1",
        unit_type=UnitType.MEDIC,
        status=UnitStatus.AVAILABLE,
        location_x=x,
        location_y=y,
        assigned_incident_id=None,
        eta_seconds=0.0,
        crew_count=2,
    )


def _incident(x: float, y: float) -> IncidentState:
    return IncidentState(
        incident_id="INC-001",
        incident_type=IncidentType.CARDIAC_ARREST,
        severity=IncidentSeverity.PRIORITY_1,
        location_x=x,
        location_y=y,
        reported_at_step=0,
        units_assigned=[],
        status=IncidentStatus.PENDING,
        survival_clock=100.0,
    )


def test_compute_eta_manhattan() -> None:
    unit = _unit(0.0, 0.0)
    inc = _incident(3.0, 4.0)
    assert compute_eta(unit, inc, unit_speed=1.0) == pytest.approx(7.0)


def test_move_unit_toward_consumes_x_then_y() -> None:
    unit = _unit(0.0, 0.0)
    inc = _incident(10.0, 10.0)
    moved = move_unit_toward(unit, inc, unit_speed=1.0, dt=5.0)
    assert moved.location_x == pytest.approx(5.0)
    assert moved.location_y == pytest.approx(0.0)


def test_check_arrival_threshold() -> None:
    unit = _unit(10.0, 10.0)
    inc = _incident(10.2, 10.1)
    assert check_arrival(unit, inc, threshold_blocks=0.5) is True


def test_coverage_score_in_range() -> None:
    units = {
        "A": _unit(1.0, 0.0),
        "B": _unit(51.0, 0.0),
        "C": _unit(99.0, 0.0),
    }
    score = compute_coverage_score(units, grid_size=(100, 100), bins_x=4)
    assert 0.0 <= score <= 1.0
