"""Dispatch protocol validation.

This module validates whether a dispatcher action is legal given the current state.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.city_schema import CitySchema
from src.models import (
    Action,
    DispatchAction,
    IncidentSeverity,
    IncidentStatus,
    State,
    UnitStatus,
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    issues: list[str]


class DispatchProtocolValidator:
    """Validates dispatch actions against the environment state."""

    @staticmethod
    def _severity_rank(sev: IncidentSeverity) -> int:
        if sev == IncidentSeverity.PRIORITY_1:
            return 3
        if sev == IncidentSeverity.PRIORITY_2:
            return 2
        return 1

    def validate(self, schema: CitySchema, state: State, action: Action) -> ValidationResult:
        issues: list[str] = []
        errors: list[str] = []

        def warn(msg: str) -> None:
            issues.append(f"warn:{msg}")

        def error(msg: str) -> None:
            issues.append(msg)
            errors.append(msg)

        unit = state.units.get(action.unit_id)
        if unit is None:
            error(f"Unknown unit_id '{action.unit_id}'")
            return ValidationResult(ok=False, issues=issues)

        incident = state.incidents.get(action.incident_id)
        if incident is None:
            error(f"Unknown incident_id '{action.incident_id}'")
            return ValidationResult(ok=False, issues=issues)

        if action.action_type in {DispatchAction.DISPATCH, DispatchAction.REASSIGN}:
            if unit.status != UnitStatus.AVAILABLE:
                error(f"Unit '{unit.unit_id}' not available (status={unit.status})")
            if unit.assigned_incident_id is not None:
                error(f"Unit '{unit.unit_id}' already assigned to '{unit.assigned_incident_id}'")
            if incident.status in {IncidentStatus.RESOLVED}:
                error(f"Incident '{incident.incident_id}' already resolved")

            # Triage type matching is a soft rule: record warning, do not invalidate.
            required = schema.default_required_units.get(incident.incident_type)
            if required is not None and required:
                if unit.unit_type not in required:
                    warn(
                        f"Unit '{unit.unit_id}' type {unit.unit_type} mismatches "
                        f"recommended {incident.incident_type} types {required}"
                    )

        elif action.action_type == DispatchAction.CANCEL:
            if unit.assigned_incident_id != incident.incident_id:
                error(f"Unit '{unit.unit_id}' not assigned to incident '{incident.incident_id}'")

        elif action.action_type == DispatchAction.MUTUAL_AID:
            if incident.status == IncidentStatus.RESOLVED:
                error(f"Incident '{incident.incident_id}' already resolved")

            # Mutual aid only when local same-type units are exhausted.
            same_type_available = any(
                u.unit_type == unit.unit_type and u.status == UnitStatus.AVAILABLE
                for u in state.units.values()
            )
            if same_type_available:
                error(
                    f"Mutual aid not allowed: local {unit.unit_type} units still AVAILABLE"
                )

        elif action.action_type in {DispatchAction.UPGRADE, DispatchAction.DOWNGRADE}:
            if incident.status == IncidentStatus.RESOLVED:
                error(f"Incident '{incident.incident_id}' already resolved")
            if action.priority_override is None:
                error("priority_override required for severity change")
            else:
                cur = self._severity_rank(incident.severity)
                new = self._severity_rank(action.priority_override)
                if action.action_type == DispatchAction.UPGRADE and new <= cur:
                    error(
                        f"UPGRADE requires higher severity than {incident.severity}"
                    )
                if action.action_type == DispatchAction.DOWNGRADE and new >= cur:
                    error(
                        f"DOWNGRADE requires lower severity than {incident.severity}"
                    )

        elif action.action_type in {
            DispatchAction.STAGE,
        }:
            # Allowed in principle; the state machine may choose to ignore.
            if incident.status == IncidentStatus.RESOLVED:
                error(f"Incident '{incident.incident_id}' already resolved")

        else:
            error(f"Unsupported action_type '{action.action_type}'")

        return ValidationResult(ok=(len(errors) == 0), issues=issues)
