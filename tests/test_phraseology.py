"""TDD tests for phraseology renderer/judge."""

import pytest
from src.models import Action, ClearanceType
from src.phraseology import PhraseologyRenderer, PhraseologyJudge


class TestPhraseologyRenderer:
    """Tests for PhraseologyRenderer.render()."""

    def test_pushback_renders_correctly(self):
        """Pushback action renders with direction."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            pushback_direction="left",
            readback_required=True,
        )
        result = renderer.render(action)
        assert result == "BAW123, pushback left, readback required"

    def test_taxi_renders_correctly(self):
        """Taxi action renders with route_join."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW456",
            route=["A", "B", "C"],
            readback_required=True,
        )
        result = renderer.render(action)
        # route_join = ", ".join(route[:-1]) + " and " + route[-1]
        # = "A, B and C"
        assert result == "BAW456, taxi to A, B and C, readback required"

    def test_taxi_single_waypoint(self):
        """Taxi with single waypoint uses that directly."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW789",
            route=["A1"],
            readback_required=True,
        )
        result = renderer.render(action)
        assert result == "BAW789, taxi to A1, readback required"

    def test_taxi_two_waypoints(self):
        """Taxi with two waypoints joins with 'and'."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW101",
            route=["A1", "B2"],
            readback_required=True,
        )
        result = renderer.render(action)
        assert result == "BAW101, taxi to A1 and B2, readback required"

    def test_hold_short_renders(self):
        """Hold short renders runway."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.HOLD_SHORT,
            target_callsign="BAW123",
            runway="27",
        )
        result = renderer.render(action)
        assert result == "BAW123, hold short of runway 27"

    def test_cross_runway_renders(self):
        """Cross runway renders runway."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.CROSS_RUNWAY,
            target_callsign="BAW123",
            runway="27",
        )
        result = renderer.render(action)
        assert result == "BAW123, cross runway 27"

    def test_line_up_renders(self):
        """Line up renders runway."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.LINE_UP,
            target_callsign="BAW123",
            runway="27",
        )
        result = renderer.render(action)
        assert result == "BAW123, line up runway 27"

    def test_takeoff_renders(self):
        """Takeoff renders runway."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign="BAW123",
            runway="27",
        )
        result = renderer.render(action)
        assert result == "BAW123, runway 27, takeoff approved"

    def test_landing_renders(self):
        """Landing renders runway."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27",
        )
        result = renderer.render(action)
        assert result == "BAW123, runway 27, you are cleared to land"

    def test_pushback_missing_direction_returns_invalid(self):
        """Pushback without direction returns INVALID."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            pushback_direction=None,
        )
        result = renderer.render(action)
        assert result.startswith("INVALID:")
        assert "direction" in result.lower()

    def test_taxi_missing_route_returns_invalid(self):
        """Taxi without route returns INVALID."""
        renderer = PhraseologyRenderer()
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=[],
        )
        result = renderer.render(action)
        assert result.startswith("INVALID:")
        assert "route" in result.lower()

    def test_runway_action_missing_runway_returns_invalid(self):
        """Runway-dependent actions without runway return INVALID."""
        renderer = PhraseologyRenderer()
        for clearance_type in [
            ClearanceType.HOLD_SHORT,
            ClearanceType.CROSS_RUNWAY,
            ClearanceType.LINE_UP,
            ClearanceType.TAKEOFF,
            ClearanceType.LANDING,
        ]:
            action = Action(
                clearance_type=clearance_type,
                target_callsign="BAW123",
                runway=None,
            )
            result = renderer.render(action)
            assert result.startswith("INVALID:"), (
                f"Expected INVALID for {clearance_type}"
            )


class TestPhraseologyJudge:
    """Tests for PhraseologyJudge.score() and check_readback()."""

    def test_exact_match_scores_one(self):
        """Exact match returns 1.0."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            pushback_direction="left",
            readback_required=True,
        )
        candidate = "BAW123, pushback left, readback required"
        score = judge.score(action, candidate)
        assert score == 1.0

    def test_partial_match_scores_half(self):
        """Partial match returns 0.5."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            pushback_direction="left",
        )
        # Contains callsign and pushback but wrong direction
        candidate = "BAW123, pushback right"
        score = judge.score(action, candidate)
        assert score == 0.5

    def test_no_match_scores_zero(self):
        """No match returns 0.0."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            pushback_direction="left",
        )
        candidate = "ABC, cleared to land runway 09"
        score = judge.score(action, candidate)
        assert score == 0.0

    def test_score_is_bounded(self):
        """Score is always in [0.0, 1.0]."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27",
        )
        for candidate in [
            "",
            "BAW123, runway 27, you are cleared to land",
            "completely wrong",
        ]:
            score = judge.score(action, candidate)
            assert 0.0 <= score <= 1.0


class TestReadbackCompleteness:
    """Tests for readback completeness checking."""

    def test_pushback_readback_needs_direction(self):
        """Pushback readback must contain direction."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign="BAW123",
            pushback_direction="left",
        )
        assert judge.check_readback("BAW123, pushback left", action) is True
        assert judge.check_readback("BAW123, pushback right", action) is False

    def test_taxi_readback_needs_route_endpoints(self):
        """Taxi readback must contain route endpoints."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=["A", "B", "C"],
        )
        # Must contain first and last waypoints
        assert judge.check_readback("BAW123, taxi to A and C", action) is True
        assert judge.check_readback("BAW123, taxi to A", action) is False
        assert judge.check_readback("BAW123, taxi to C", action) is False

    def test_hold_short_readback_needs_runway(self):
        """Hold short readback must contain runway."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.HOLD_SHORT,
            target_callsign="BAW123",
            runway="27",
        )
        assert judge.check_readback("BAW123, hold short of runway 27", action) is True
        assert judge.check_readback("BAW123, hold short of runway 09", action) is False

    def test_line_up_readback_needs_runway(self):
        """Line up readback must contain runway."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.LINE_UP,
            target_callsign="BAW123",
            runway="27",
        )
        assert judge.check_readback("BAW123, line up runway 27", action) is True
        assert judge.check_readback("BAW123, line up runway 09", action) is False

    def test_takeoff_readback_needs_runway(self):
        """Takeoff readback must contain runway."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign="BAW123",
            runway="27",
        )
        assert (
            judge.check_readback("BAW123, runway 27, takeoff approved", action) is True
        )
        assert (
            judge.check_readback("BAW123, runway 09, takeoff approved", action) is False
        )

    def test_landing_readback_needs_runway(self):
        """Landing readback must contain runway."""
        judge = PhraseologyJudge()
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27",
        )
        assert (
            judge.check_readback("BAW123, runway 27, you are cleared to land", action)
            is True
        )
        assert (
            judge.check_readback("BAW123, runway 09, you are cleared to land", action)
            is False
        )
