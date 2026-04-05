"""Tests proving the TDD scaffold test harness itself works."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_conftest_fixtures_available(
    seeded_random,
    sample_aircraft_state,
    sample_action,
    gatwick_schema,
    dummy_schema,
) -> None:
    """All conftest fixtures are accessible and return expected types."""
    import random

    from src.airport_schema import AirportSchema
    from src.models import Action, AircraftState

    assert isinstance(seeded_random, random.Random)
    expected = random.Random(42).randint(1, 100)
    actual = seeded_random.randint(1, 100)
    assert actual == expected, "seeded_random should produce deterministic values"

    assert isinstance(sample_aircraft_state, dict)
    AircraftState(**sample_aircraft_state)

    assert isinstance(sample_action, dict)
    Action(**sample_action)

    assert isinstance(gatwick_schema, AirportSchema)
    assert isinstance(dummy_schema, AirportSchema)


def test_helpers_importable() -> None:
    """Helper functions are importable and have expected signatures."""
    from tests.helpers import assert_invalid_model, assert_valid_model, capture_stdout

    assert callable(capture_stdout)
    assert callable(assert_valid_model)
    assert callable(assert_invalid_model)

    def dummy_func() -> None:
        print("hello")

    result = capture_stdout(dummy_func)
    assert result.strip() == "hello"


def test_pytest_runs_clean() -> None:
    """Scaffold tests can be collected without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "--collect-only", "-q"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Collection failed: {result.stderr}"
