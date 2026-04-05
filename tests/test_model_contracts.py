"""Model contract tests for the 911 dispatch domain."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

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


class ModelContractTests(unittest.TestCase):
    def test_dispatch_action_enum(self) -> None:
        self.assertIn(DispatchAction.DISPATCH, set(DispatchAction))

    def test_action_round_trip(self) -> None:
        action = Action(
            action_type=DispatchAction.DISPATCH,
            unit_id="MED-1",
            incident_id="INC-001",
            notes=None,
            priority_override=None,
        )
        dumped = action.model_dump()
        self.assertEqual(Action.model_validate(dumped), action)
        self.assertEqual(Action.model_validate_json(action.model_dump_json()), action)

    def test_observation_round_trip(self) -> None:
        obs = Observation(result="ok", score=0.5, protocol_ok=True, issues=[])
        self.assertEqual(Observation.model_validate(obs.model_dump()), obs)

    def test_unit_state_round_trip(self) -> None:
        unit = UnitState(
            unit_id="MED-1",
            unit_type=UnitType.MEDIC,
            status=UnitStatus.AVAILABLE,
            location_x=1.0,
            location_y=2.0,
            assigned_incident_id=None,
            eta_seconds=0.0,
            crew_count=2,
        )
        self.assertEqual(UnitState.model_validate(unit.model_dump()), unit)

    def test_incident_state_round_trip(self) -> None:
        inc = IncidentState(
            incident_id="INC-001",
            incident_type=IncidentType.CARDIAC_ARREST,
            severity=IncidentSeverity.PRIORITY_1,
            location_x=1.0,
            location_y=2.0,
            reported_at_step=0,
            units_assigned=[],
            status=IncidentStatus.PENDING,
            survival_clock=100.0,
        )
        self.assertEqual(IncidentState.model_validate(inc.model_dump()), inc)

    def test_state_round_trip(self) -> None:
        state = State(
            units={},
            incidents={},
            episode_id="ep",
            step_count=0,
            task_id="single_incident",
            city_time=0.0,
            metadata={},
        )
        self.assertEqual(State.model_validate(state.model_dump()), state)

    def test_extra_fields_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Action.model_validate(
                {
                    "action_type": "DISPATCH",
                    "unit_id": "MED-1",
                    "incident_id": "INC-001",
                    "extra": True,
                }
            )
