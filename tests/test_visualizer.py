"""Tests for the 2D dispatch visualizer."""

from __future__ import annotations

from pathlib import Path

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
from src.visualizer.viewer import Viewer2D


def _state() -> State:
    unit = UnitState(
        unit_id="MED-1",
        unit_type=UnitType.MEDIC,
        status=UnitStatus.AVAILABLE,
        location_x=10.0,
        location_y=10.0,
        assigned_incident_id=None,
        eta_seconds=0.0,
        crew_count=2,
    )
    inc = IncidentState(
        incident_id="INC-001",
        incident_type=IncidentType.CARDIAC_ARREST,
        severity=IncidentSeverity.PRIORITY_1,
        location_x=12.0,
        location_y=14.0,
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
        metadata={"grid_size": [100, 100]},
    )


def test_update_syncs_fields() -> None:
    viewer = Viewer2D()
    st = _state()
    viewer.update(st)
    assert viewer.step_count == 0
    assert viewer.task_id == "single_incident"
    assert "MED-1" in viewer.units


def test_render_png_bytes() -> None:
    viewer = Viewer2D()
    viewer.update(_state())
    png = viewer.render()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_to_file(tmp_path: Path) -> None:
    viewer = Viewer2D()
    out = tmp_path / "frame.png"
    viewer.render_to_file(str(out), _state())
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
