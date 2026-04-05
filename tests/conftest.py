"""Shared pytest fixtures for 911 dispatch supervisor test suite."""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.city_schema import CitySchema
from src.models import (
    Action,
    DispatchAction,
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    UnitStatus,
    UnitType,
)


@pytest.fixture
def seeded_random() -> random.Random:
    """Return a random.Random instance seeded with 42 for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def sample_unit_state() -> dict[str, Any]:
    """Return a minimal valid UnitState dict."""
    return {
        "unit_id": "MED-1",
        "unit_type": UnitType.MEDIC,
        "status": UnitStatus.AVAILABLE,
        "location_x": 10.0,
        "location_y": 10.0,
        "assigned_incident_id": None,
        "eta_seconds": 0.0,
        "crew_count": 2,
    }


@pytest.fixture
def sample_incident_state() -> dict[str, Any]:
    """Return a minimal valid IncidentState dict."""
    return {
        "incident_id": "INC-001",
        "incident_type": IncidentType.CARDIAC_ARREST,
        "severity": IncidentSeverity.PRIORITY_1,
        "location_x": 12.0,
        "location_y": 14.0,
        "reported_at_step": 0,
        "units_assigned": [],
        "status": IncidentStatus.PENDING,
        "survival_clock": 240.0,
    }


@pytest.fixture
def sample_action() -> dict[str, Any]:
    """Return a minimal valid dispatch Action dict."""
    return {
        "action_type": DispatchAction.DISPATCH,
        "unit_id": "MED-1",
        "incident_id": "INC-001",
        "notes": None,
        "priority_override": None,
    }


@pytest.fixture
def metro_city_schema() -> CitySchema:
    """Return a minimal valid CitySchema instance."""
    return CitySchema(
        city_name="Metro City",
        grid_size=[100, 100],
        districts=["downtown", "northside"],
        units=[],
        unit_speeds={
            UnitType.MEDIC: 1.0,
            UnitType.ENGINE: 0.8,
        },
        default_required_units={
            IncidentType.CARDIAC_ARREST: [UnitType.MEDIC],
            IncidentType.STRUCTURE_FIRE: [UnitType.ENGINE],
        },
    )
