"""Tests for the normalized hybrid ATC protocol layer."""

import pytest

from src.protocol import (
    ClearanceDefinition,
    ClearanceType,
    HandoffProtocol,
    ProtocolValidator,
)


class TestClearanceType:
    """Tests for ClearanceType StrEnum values."""

    def test_clearance_type_values(self):
        assert ClearanceType.PUSHBACK == "pushback"
        assert ClearanceType.TAXI == "taxi"
        assert ClearanceType.HOLD_SHORT == "hold_short"
        assert ClearanceType.CROSS_RUNWAY == "cross_runway"
        assert ClearanceType.LINE_UP == "line_up"
        assert ClearanceType.TAKEOFF == "takeoff"
        assert ClearanceType.LANDING == "landing"

    def test_clearance_type_is_str_enum(self):
        from enum import StrEnum

        assert issubclass(ClearanceType, StrEnum)


class TestClearanceDefinition:
    """Tests for ClearanceDefinition structure."""

    def test_clearance_definition_fields(self):
        cd = ClearanceDefinition(
            type=ClearanceType.PUSHBACK,
            required_fields=["pushback_direction"],
            phraseology_template="{callsign}, pushback {direction} approved",
            readback_required=True,
        )
        assert cd.type == ClearanceType.PUSHBACK
        assert cd.required_fields == ["pushback_direction"]
        assert cd.readback_required is True

    def test_clearance_definition_taxi(self):
        cd = ClearanceDefinition(
            type=ClearanceType.TAXI,
            required_fields=["route"],
            phraseology_template="{callsign}, taxi to {destination}",
            readback_required=True,
        )
        assert cd.type == ClearanceType.TAXI
        assert cd.required_fields == ["route"]


class TestProtocolValidator:
    """Tests for ProtocolValidator."""

    def setup_validator(self):
        return ProtocolValidator()

    def test_validate_pushback_missing_direction_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            route=[],
            pushback_direction=None,
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None
        assert "pushback_direction" in error

    def test_validate_pushback_with_direction_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            route=[],
            pushback_direction="left",
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_validate_taxi_empty_route_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=[],
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None
        assert "route" in error

    def test_validate_taxi_with_route_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=["A1", "A2", "B3"],
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_validate_hold_short_missing_runway_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.HOLD_SHORT,
            target_callsign="BAW123",
            route=[],
            runway=None,
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None
        assert "runway" in error

    def test_validate_hold_short_with_runway_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.HOLD_SHORT,
            target_callsign="BAW123",
            route=[],
            runway="27L",
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_validate_cross_runway_missing_runway_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.CROSS_RUNWAY,
            target_callsign="BAW123",
            route=[],
            runway=None,
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None

    def test_validate_cross_runway_with_runway_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.CROSS_RUNWAY,
            target_callsign="BAW123",
            route=[],
            runway="27L",
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_validate_line_up_missing_runway_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.LINE_UP,
            target_callsign="BAW123",
            route=[],
            runway=None,
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None

    def test_validate_line_up_with_runway_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.LINE_UP,
            target_callsign="BAW123",
            route=[],
            runway="27L",
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_validate_takeoff_missing_runway_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign="BAW123",
            route=[],
            runway=None,
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None

    def test_validate_takeoff_with_runway_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign="BAW123",
            route=[],
            runway="27L",
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_validate_landing_missing_runway_invalid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            route=[],
            runway=None,
        )
        valid, error = validator.validate_clearance(action)
        assert valid is False
        assert error is not None

    def test_validate_landing_with_runway_valid(self):
        validator = self.setup_validator()
        from src.models import Action

        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            route=[],
            runway="27L",
        )
        valid, error = validator.validate_clearance(action)
        assert valid is True
        assert error is None

    def test_is_valid_route_segment_taxiway_to_taxiway(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("taxiway", "taxiway", "taxi") is True

    def test_is_valid_route_segment_taxiway_to_runway(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("taxiway", "runway", "cross") is True

    def test_is_valid_route_segment_runway_to_taxiway(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("runway", "taxiway", "exit") is True

    def test_is_valid_route_segment_runway_to_runway_takeoff(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("runway", "runway", "takeoff") is True

    def test_is_valid_route_segment_runway_to_runway_landing(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("runway", "runway", "landing") is True

    def test_is_valid_route_segment_taxiway_to_runway_taxi_invalid(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("taxiway", "runway", "taxi") is False

    def test_is_valid_route_segment_unknown_from_node_invalid(self):
        validator = self.setup_validator()
        assert validator.is_valid_route_segment("unknown", "taxiway", "taxi") is False


class TestHandoffProtocol:
    """Tests for HandoffProtocol."""

    def setup_handoff(self):
        return HandoffProtocol()

    def test_frequencies_values(self):
        assert HandoffProtocol.FREQUENCIES["ground"] == 121.9
        assert HandoffProtocol.FREQUENCIES["tower"] == 118.9
        assert HandoffProtocol.FREQUENCIES["approach"] == 119.9
        assert HandoffProtocol.FREQUENCIES["departure"] == 132.6

    def test_validate_handoff_ground_to_tower_valid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(121.9, 118.9, "taxi_out") is True

    def test_validate_handoff_tower_to_departure_valid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(118.9, 132.6, "takeoff") is True

    def test_validate_handoff_approach_to_tower_valid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(119.9, 118.9, "landing") is True

    def test_validate_handoff_invalid_from_freq_invalid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(999.0, 118.9, "taxi_out") is False

    def test_validate_handoff_invalid_to_freq_invalid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(121.9, 999.0, "taxi_out") is False

    def test_validate_handoff_no_entry_for_phase_invalid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(121.9, 118.9, "unknown_phase") is False

    def test_validate_handoff_same_freq_invalid(self):
        handoff = self.setup_handoff()
        assert handoff.validate_handoff(121.9, 121.9, "taxi_out") is False
