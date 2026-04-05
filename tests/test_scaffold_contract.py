"""Tests proving the TDD scaffold test harness itself works (dispatch edition)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.city_schema import CitySchema
from src.models import Action, IncidentState, UnitState

ROOT = Path(__file__).resolve().parents[1]


def test_conftest_fixtures_available(
    seeded_random,
    sample_unit_state,
    sample_incident_state,
    sample_action,
    metro_city_schema,
) -> None:
    import random

    assert isinstance(seeded_random, random.Random)
    expected = random.Random(42).randint(1, 100)
    actual = seeded_random.randint(1, 100)
    assert actual == expected

    UnitState(**sample_unit_state)
    IncidentState(**sample_incident_state)
    Action(**sample_action)
    assert isinstance(metro_city_schema, CitySchema)


def test_helpers_importable() -> None:
    from tests.helpers import assert_invalid_model, assert_valid_model, capture_stdout

    assert callable(capture_stdout)
    assert callable(assert_valid_model)
    assert callable(assert_invalid_model)


def test_pytest_collects_this_file() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"Collection failed: {result.stderr}"
