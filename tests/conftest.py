"""Shared pytest fixtures for ATC Ground Control test suite."""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.airport_schema import (
    AirportEdge,
    AirportNode,
    AirportSchema,
    EdgeMovement,
    NodeType,
)
from src.models import (
    Action,
    AircraftState,
    ClearanceType,
    LifecyclePhase,
)


@pytest.fixture
def seeded_random() -> random.Random:
    """Return a random.Random instance seeded with 42 for deterministic tests."""
    return random.Random(42)


@pytest.fixture
def sample_aircraft_state() -> dict[str, Any]:
    """Return a minimal valid AircraftState dict."""
    return {
        "callsign": "BAW123",
        "x_ft": 0.0,
        "y_ft": 0.0,
        "heading_deg": 90.0,
        "altitude_ft": 0.0,
        "speed_kt": 0.0,
        "phase": LifecyclePhase.ARRIVAL_HANDOFF,
        "assigned_runway": None,
        "assigned_gate": None,
        "wake_category": "M",
    }


@pytest.fixture
def sample_action() -> dict[str, Any]:
    """Return a minimal valid Action dict."""
    return {
        "clearance_type": ClearanceType.TAXI,
        "target_callsign": "BAW123",
        "route": [],
        "readback_required": False,
        "pushback_direction": None,
        "hold_short": False,
        "runway": None,
    }


@pytest.fixture
def gatwick_schema() -> AirportSchema:
    """Return a minimal valid Gatwick AirportSchema instance."""
    return AirportSchema(
        airport_code="EGKK",
        nodes={
            "stand_1": AirportNode(
                id="stand_1",
                node_type=NodeType.STAND,
                x_ft=0.0,
                y_ft=0.0,
            )
        },
        edges=[
            AirportEdge(
                from_node="stand_1",
                to_node="stand_1",
                movement_type=EdgeMovement.TAXI,
                distance_ft=100.0,
                max_speed_kt=20.0,
            )
        ],
        runways=[],
        gates=[],
    )


@pytest.fixture
def dummy_schema() -> AirportSchema:
    """Return a minimal valid dummy AirportSchema instance."""
    return AirportSchema(
        airport_code="DUMMY",
        nodes={
            "stand_1": AirportNode(
                id="stand_1",
                node_type=NodeType.STAND,
                x_ft=0.0,
                y_ft=0.0,
            )
        },
        edges=[
            AirportEdge(
                from_node="stand_1",
                to_node="stand_1",
                movement_type=EdgeMovement.TAXI,
                distance_ft=100.0,
                max_speed_kt=20.0,
            )
        ],
        runways=[],
        gates=[],
    )
