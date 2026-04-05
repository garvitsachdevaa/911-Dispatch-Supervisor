# pyright: reportMissingImports=false

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from src.models import (
    Action,
    AircraftState,
    ClearanceType,
    LifecyclePhase,
    Observation,
    State,
)


class ModelContractTests(unittest.TestCase):
    def test_lifecycle_phase_enum_exhaustiveness(self) -> None:
        phases = set(LifecyclePhase)
        expected = {
            "approach",
            "landing",
            "arrival_handoff",
            "taxi_in",
            "docking",
            "at_gate",
            "pushback",
            "taxi_out",
            "departure_queue",
            "takeoff",
            "departed",
        }
        self.assertEqual(phases, expected)

    def test_clearance_type_enum(self) -> None:
        clearances = set(ClearanceType)
        expected = {
            "pushback",
            "taxi",
            "hold_short",
            "cross_runway",
            "line_up",
            "takeoff",
            "landing",
        }
        self.assertEqual(clearances, expected)

    def test_action_round_trip_serialization(self) -> None:
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=["alpha", "bravo", "charlie"],
            readback_required=True,
            pushback_direction=None,
            hold_short=False,
            runway=None,
        )
        dumped = action.model_dump()
        self.assertEqual(Action.model_validate(dumped), action)
        self.assertEqual(Action.model_validate_json(action.model_dump_json()), action)

    def test_action_minimal_round_trip(self) -> None:
        action = Action(
            clearance_type=ClearanceType.HOLD_SHORT,
            target_callsign="EZY456",
            route=[],
            readback_required=False,
            hold_short=True,
        )
        self.assertEqual(Action.model_validate(action.model_dump()), action)

    def test_observation_round_trip_serialization(self) -> None:
        observation = Observation(
            result="Cleared to land",
            score=1.0,
            phraseology_ok=True,
            issues=[],
        )
        self.assertEqual(
            Observation.model_validate(observation.model_dump()), observation
        )
        self.assertEqual(
            Observation.model_validate_json(observation.model_dump_json()), observation
        )

    def test_aircraft_state_round_trip_serialization(self) -> None:
        aircraft = AircraftState(
            callsign="BAW789",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=270.0,
            altitude_ft=35000.0,
            speed_kt=250.0,
            phase=LifecyclePhase.APPROACH,
            assigned_runway="27L",
            assigned_gate=None,
            wake_category="H",
        )
        self.assertEqual(AircraftState.model_validate(aircraft.model_dump()), aircraft)

    def test_state_round_trip_serialization(self) -> None:
        aircraft = AircraftState(
            callsign="BAW123",
            x_ft=100.0,
            y_ft=200.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.TAXI_IN,
            assigned_runway=None,
            assigned_gate="A12",
            wake_category="M",
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW123": aircraft},
            episode_id="ep-001",
            step_count=10,
            task_id="arrival",
            metadata={},
        )
        self.assertEqual(State.model_validate(state.model_dump()), state)
        self.assertEqual(State.model_validate_json(state.model_dump_json()), state)

    def test_invalid_enum_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AircraftState.model_validate(
                {
                    "callsign": "BAW123",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 0.0,
                    "phase": "not_a_phase",
                    "assigned_runway": None,
                    "assigned_gate": None,
                    "wake_category": "M",
                }
            )

    def test_extra_fields_rejected_action(self) -> None:
        with self.assertRaises(ValidationError):
            Action.model_validate(
                {
                    "clearance_type": "taxi",
                    "target_callsign": "BAW123",
                    "route": [],
                    "extra_field": True,
                }
            )

    def test_extra_fields_rejected_observation(self) -> None:
        with self.assertRaises(ValidationError):
            Observation.model_validate(
                {
                    "result": "ok",
                    "score": 1.0,
                    "phraseology_ok": True,
                    "issues": [],
                    "unexpected": "field",
                }
            )

    def test_extra_fields_rejected_aircraft_state(self) -> None:
        with self.assertRaises(ValidationError):
            AircraftState.model_validate(
                {
                    "callsign": "BAW123",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 0.0,
                    "phase": "taxi_in",
                    "unknown_field": True,
                }
            )

    def test_extra_fields_rejected_state(self) -> None:
        with self.assertRaises(ValidationError):
            State.model_validate(
                {
                    "phase": "approach",
                    "aircraft": {},
                    "episode_id": "ep-001",
                    "step_count": 0,
                    "task_id": "arrival",
                    "metadata": {},
                    "bad_field": True,
                }
            )

    def test_heading_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AircraftState.model_validate(
                {
                    "callsign": "BAW123",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 400.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 0.0,
                    "phase": "approach",
                    "assigned_runway": None,
                    "assigned_gate": None,
                    "wake_category": "M",
                }
            )

    def test_altitude_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AircraftState.model_validate(
                {
                    "callsign": "BAW123",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 50000.0,
                    "speed_kt": 0.0,
                    "phase": "approach",
                    "assigned_runway": None,
                    "assigned_gate": None,
                    "wake_category": "M",
                }
            )

    def test_speed_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AircraftState.model_validate(
                {
                    "callsign": "BAW123",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 700.0,
                    "phase": "taxi_in",
                    "assigned_runway": None,
                    "assigned_gate": None,
                    "wake_category": "M",
                }
            )

    def test_score_out_of_range_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            Observation.model_validate(
                {
                    "result": "ok",
                    "score": 1.5,
                    "phraseology_ok": True,
                    "issues": [],
                }
            )

    def test_empty_callsign_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AircraftState.model_validate(
                {
                    "callsign": "",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 0.0,
                    "phase": "approach",
                    "assigned_runway": None,
                    "assigned_gate": None,
                    "wake_category": "M",
                }
            )

    def test_state_multiple_aircraft(self) -> None:
        aircraft1 = AircraftState(
            callsign="BAW123",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.TAXI_IN,
            assigned_runway=None,
            assigned_gate="A1",
            wake_category="M",
        )
        aircraft2 = AircraftState(
            callsign="EZY456",
            x_ft=100.0,
            y_ft=200.0,
            heading_deg=180.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.TAXI_IN,
            assigned_runway=None,
            assigned_gate="B2",
            wake_category="M",
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW123": aircraft1, "EZY456": aircraft2},
            episode_id="ep-002",
            step_count=5,
            task_id="arrival",
            metadata={"controller": "ground"},
        )
        self.assertEqual(State.model_validate(state.model_dump()), state)
        self.assertEqual(len(state.aircraft), 2)

    def test_action_pushback_direction_required_for_pushback(self) -> None:
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            route=[],
            readback_required=True,
            pushback_direction="north",
            hold_short=False,
            runway=None,
        )
        self.assertEqual(action.pushback_direction, "north")


if __name__ == "__main__":
    unittest.main()
