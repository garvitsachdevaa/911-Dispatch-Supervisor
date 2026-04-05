"""Dispatch protocol validation.

This module validates whether a dispatcher action is legal given the current state.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.city_schema import CitySchema
from src.models import Action, DispatchAction, IncidentStatus, State, UnitStatus


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[str]


class DispatchProtocolValidator:
    """Validates dispatch actions against the environment state."""

    def validate(self, schema: CitySchema, state: State, action: Action) -> ValidationResult:
        issues: list[str] = []

        unit = state.units.get(action.unit_id)
        if unit is None:
            issues.append(f"Unknown unit_id '{action.unit_id}'")
            return ValidationResult(ok=False, issues=issues)

        incident = state.incidents.get(action.incident_id)
        if incident is None:
            issues.append(f"Unknown incident_id '{action.incident_id}'")
            return ValidationResult(ok=False, issues=issues)

        if action.action_type == DispatchAction.DISPATCH:
            if unit.status != UnitStatus.AVAILABLE:
                issues.append(f"Unit '{unit.unit_id}' not available (status={unit.status})")
            if unit.assigned_incident_id is not None:
                issues.append(
                    f"Unit '{unit.unit_id}' already assigned to '{unit.assigned_incident_id}'"
                )
            if incident.status in {IncidentStatus.RESOLVED}:
                issues.append(f"Incident '{incident.incident_id}' already resolved")

        elif action.action_type == DispatchAction.CANCEL:
            if unit.assigned_incident_id != incident.incident_id:
                issues.append(
                    f"Unit '{unit.unit_id}' not assigned to incident '{incident.incident_id}'"
                )

        elif action.action_type in {
            DispatchAction.UPGRADE,
            DispatchAction.DOWNGRADE,
            DispatchAction.MUTUAL_AID,
            DispatchAction.REASSIGN,
            DispatchAction.STAGE,
        }:
            # Allowed in principle; the state machine may choose to ignore.
            if incident.status == IncidentStatus.RESOLVED:
                issues.append(f"Incident '{incident.incident_id}' already resolved")

        else:
            issues.append(f"Unsupported action_type '{action.action_type}'")

        return ValidationResult(ok=(len(issues) == 0), issues=issues)
