"""Renderer/judge for normalized ATC phraseology subset."""

from pydantic import BaseModel

from src.models import Action, ClearanceType


class PhraseologyRenderer(BaseModel):
    """Converts structured actions to standardized phraseology strings."""

    def _build_route_join(self, route: list[str]) -> str:
        """Build human-readable route string."""
        if len(route) == 0:
            return ""
        if len(route) == 1:
            return route[0]
        if len(route) == 2:
            return f"{route[0]} and {route[1]}"
        return f"{', '.join(route[:-1])} and {route[-1]}"

    def _render_pushback(self, action: Action) -> str:
        if not action.pushback_direction:
            return f"INVALID: pushback requires direction"
        base = f"{action.target_callsign}, pushback {action.pushback_direction}"
        if action.readback_required:
            base += ", readback required"
        return base

    def _render_taxi(self, action: Action) -> str:
        if not action.route:
            return f"INVALID: taxi requires route"
        route_join = self._build_route_join(action.route)
        base = f"{action.target_callsign}, taxi to {route_join}"
        if action.readback_required:
            base += ", readback required"
        return base

    def _render_hold_short(self, action: Action) -> str:
        if not action.runway:
            return f"INVALID: hold_short requires runway"
        return f"{action.target_callsign}, hold short of runway {action.runway}"

    def _render_cross_runway(self, action: Action) -> str:
        if not action.runway:
            return f"INVALID: cross_runway requires runway"
        return f"{action.target_callsign}, cross runway {action.runway}"

    def _render_line_up(self, action: Action) -> str:
        if not action.runway:
            return f"INVALID: line_up requires runway"
        return f"{action.target_callsign}, line up runway {action.runway}"

    def _render_takeoff(self, action: Action) -> str:
        if not action.runway:
            return f"INVALID: takeoff requires runway"
        return f"{action.target_callsign}, runway {action.runway}, takeoff approved"

    def _render_landing(self, action: Action) -> str:
        if not action.runway:
            return f"INVALID: landing requires runway"
        return (
            f"{action.target_callsign}, runway {action.runway}, you are cleared to land"
        )

    def render(self, action: Action) -> str:
        """Convert structured Action to normalized phraseology string."""
        dispatch = {
            ClearanceType.PUSHBACK: self._render_pushback,
            ClearanceType.TAXI: self._render_taxi,
            ClearanceType.HOLD_SHORT: self._render_hold_short,
            ClearanceType.CROSS_RUNWAY: self._render_cross_runway,
            ClearanceType.LINE_UP: self._render_line_up,
            ClearanceType.TAKEOFF: self._render_takeoff,
            ClearanceType.LANDING: self._render_landing,
        }
        renderer = dispatch.get(action.clearance_type)
        if not renderer:
            return f"INVALID: unknown clearance type {action.clearance_type}"
        return renderer(action)


class PhraseologyJudge(BaseModel):
    """Scores candidate phraseology against structured action truth."""

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text into punctuation-stripped lowercase words."""
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return set(tokens)

    def _get_canonical(self, action: Action) -> str:
        """Get canonical rendered phraseology for action."""
        renderer = PhraseologyRenderer()
        return renderer.render(action)

    def score(self, ground_truth_action: Action, candidate_text: str) -> float:
        """
        Score candidate phraseology against ground truth action.
        Returns 1.0 for exact match, 0.5 for partial, 0.0 for no match.
        """
        canonical = self._get_canonical(ground_truth_action)
        norm_candidate = candidate_text.lower().strip()
        norm_canonical = canonical.lower().strip()

        if norm_candidate == norm_canonical:
            return 1.0

        canonical_tokens = self._tokenize(canonical)
        candidate_tokens = self._tokenize(candidate_text)

        overlap = canonical_tokens & candidate_tokens

        if not overlap:
            return 0.0

        overlap_ratio = len(overlap) / len(canonical_tokens)

        if overlap_ratio >= 0.5:
            return 0.5
        return 0.0

    def check_readback(self, candidate_text: str, ground_truth_action: Action) -> bool:
        """Check if candidate contains required readback elements."""
        candidate_tokens = self._tokenize(candidate_text)

        if ground_truth_action.clearance_type == ClearanceType.PUSHBACK:
            direction = ground_truth_action.pushback_direction
            if not direction:
                return False
            return direction.lower() in candidate_tokens

        if ground_truth_action.clearance_type == ClearanceType.TAXI:
            route = ground_truth_action.route
            if not route or len(route) < 1:
                return False
            first = route[0].lower()
            last = route[-1].lower()
            return first in candidate_tokens and last in candidate_tokens

        runway = ground_truth_action.runway
        if not runway:
            return False
        return runway.lower() in candidate_tokens
