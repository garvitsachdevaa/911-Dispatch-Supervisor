"""Renderer/judge for normalized dispatch phraseology.

This repository originally shipped an ATC phraseology module. After the domain
pivot to 911 dispatch, we keep the same public symbols (`PhraseologyRenderer`,
`PhraseologyJudge`) but implement dispatch-oriented rendering and scoring.
"""

from __future__ import annotations

from pydantic import BaseModel

from src.models import Action, DispatchAction


class PhraseologyRenderer(BaseModel):
    """Converts structured dispatch actions to standardized strings."""

    def render(self, action: Action) -> str:
        if action.action_type == DispatchAction.DISPATCH:
            return f"DISPATCH {action.unit_id} -> {action.incident_id}"
        if action.action_type == DispatchAction.REASSIGN:
            return f"REASSIGN {action.unit_id} -> {action.incident_id}"
        if action.action_type == DispatchAction.CANCEL:
            return f"CANCEL {action.unit_id} FROM {action.incident_id}"
        if action.action_type == DispatchAction.STAGE:
            return f"STAGE {action.unit_id} FOR {action.incident_id}"
        if action.action_type == DispatchAction.MUTUAL_AID:
            return f"MUTUAL_AID {action.unit_id} -> {action.incident_id}"
        if action.action_type in {DispatchAction.UPGRADE, DispatchAction.DOWNGRADE}:
            if action.priority_override is None:
                return f"INVALID: {action.action_type.value} requires priority_override"
            return (
                f"{action.action_type.value} {action.incident_id} "
                f"TO {action.priority_override.value}"
            )
        return f"INVALID: unknown action_type {action.action_type}"


class PhraseologyJudge(BaseModel):
    """Scores candidate phraseology against structured action truth."""

    def _tokenize(self, text: str) -> set[str]:
        import re

        # Treat common dispatch IDs like "MED-1" and "INC-001" as single tokens.
        return set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()))

    def score(self, ground_truth_action: Action, candidate_text: str) -> float:
        canonical = PhraseologyRenderer().render(ground_truth_action)
        norm_candidate = candidate_text.strip().lower()
        norm_canonical = canonical.strip().lower()

        if norm_candidate == norm_canonical:
            return 1.0

        canonical_tokens = self._tokenize(canonical)
        candidate_tokens = self._tokenize(candidate_text)
        overlap = canonical_tokens & candidate_tokens
        if not overlap:
            return 0.0
        return 0.5 if (len(overlap) / max(len(canonical_tokens), 1)) >= 0.5 else 0.0

    def check_readback(self, candidate_text: str, ground_truth_action: Action) -> bool:
        tokens = self._tokenize(candidate_text)

        if ground_truth_action.action_type in {DispatchAction.DISPATCH, DispatchAction.REASSIGN}:
            return (
                ground_truth_action.unit_id.lower() in tokens
                and ground_truth_action.incident_id.lower() in tokens
            )

        if ground_truth_action.action_type == DispatchAction.CANCEL:
            return (
                ground_truth_action.unit_id.lower() in tokens
                and ground_truth_action.incident_id.lower() in tokens
            )

        if ground_truth_action.action_type in {DispatchAction.UPGRADE, DispatchAction.DOWNGRADE}:
            if ground_truth_action.priority_override is None:
                return False
            return (
                ground_truth_action.incident_id.lower() in tokens
                and ground_truth_action.priority_override.value.lower() in tokens
            )

        return True
