"""Dispatch state machine for the 911 supervisor environment."""

from __future__ import annotations

import math
import random

from src.city_schema import CitySchema
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
)
from src.protocol import DispatchProtocolValidator
from src.tasks.registry import DispatchScenarioFactory

DEFAULT_DT_S = 30.0
MAX_STEPS = 200


def _severity_deadline_seconds(severity: IncidentSeverity) -> float:
    if severity == IncidentSeverity.PRIORITY_1:
        return 600.0
    if severity == IncidentSeverity.PRIORITY_2:
        return 1200.0
    return 1800.0


def _resolve_timer_seconds(severity: IncidentSeverity) -> float:
    if severity == IncidentSeverity.PRIORITY_1:
        return 300.0
    if severity == IncidentSeverity.PRIORITY_2:
        return 600.0
    return 900.0


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


class DispatchStateMachine:
    """Deterministic dispatch state machine.

    Supports a minimal action set (DISPATCH, CANCEL) and advances incidents through:
    PENDING → RESPONDING → ON_SCENE → RESOLVED.
    """

    def __init__(self, schema: CitySchema, seed: int | None = None) -> None:
        self._schema = schema
        self._seed = seed
        self._rng = random.Random(seed)
        self._validator = DispatchProtocolValidator()
        self._incident_counter = 0

    def reset(self, task_id: str, episode_id: str) -> State:
        self._rng = random.Random(self._seed)
        self._incident_counter = 0

        seed = self._seed if self._seed is not None else 42
        state_dict, meta = DispatchScenarioFactory.build(task_id=task_id, seed=seed)
        state_dict["episode_id"] = episode_id

        state = State.model_validate(state_dict)

        # Enrich metadata with schema-derived info for rewards and validation.
        schema_dump = self._schema.model_dump()
        state.metadata.setdefault("seed", seed)
        state.metadata.setdefault("schema", self._schema.city_name)
        state.metadata.setdefault("districts", meta.get("districts", schema_dump.get("districts", [])))
        state.metadata.setdefault("grid_size", meta.get("grid_size", schema_dump.get("grid_size", [])))
        state.metadata.setdefault("unit_speeds", schema_dump.get("unit_speeds", {}))
        state.metadata.setdefault(
            "default_required_units", schema_dump.get("default_required_units", {})
        )

        state.metadata["max_steps"] = int(meta.get("max_steps", MAX_STEPS))
        state.metadata["waves"] = list(meta.get("waves", []))
        state.metadata["unit_status_changes"] = list(meta.get("unit_status_changes", []))
        if "mutual_aid_eta_penalty" in meta:
            state.metadata["mutual_aid_eta_penalty"] = float(meta["mutual_aid_eta_penalty"])

        state.metadata.setdefault("resolved_incidents", [])
        state.metadata.setdefault("failed_incidents", [])
        state.metadata.setdefault("p1_seen", [])

        # Apply any wave configured for step 0 at reset.
        for wave in list(state.metadata.get("waves", [])):
            if int(wave.get("at_step", -1)) != 0:
                continue
            for inc in wave.get("incidents", []):
                incident_obj = IncidentState.model_validate(inc)
                state.incidents[incident_obj.incident_id] = incident_obj

        # Initialize P1 tracking.
        for inc in state.incidents.values():
            if inc.severity == IncidentSeverity.PRIORITY_1 and inc.incident_id not in state.metadata["p1_seen"]:
                state.metadata["p1_seen"].append(inc.incident_id)

        return state

    def get_legal_actions(self, state: State) -> list[Action]:
        actions: list[Action] = []

        active_incidents = [
            i
            for i in state.incidents.values()
            if i.status not in {IncidentStatus.RESOLVED}
        ]
        if not active_incidents:
            return actions

        for unit in state.units.values():
            if unit.status == UnitStatus.AVAILABLE:
                for incident in active_incidents:
                    actions.append(
                        Action(
                            action_type=DispatchAction.DISPATCH,
                            unit_id=unit.unit_id,
                            incident_id=incident.incident_id,
                        )
                    )
            elif unit.assigned_incident_id is not None:
                actions.append(
                    Action(
                        action_type=DispatchAction.CANCEL,
                        unit_id=unit.unit_id,
                        incident_id=unit.assigned_incident_id,
                    )
                )
        return actions

    def step(self, state: State, action: Action) -> tuple[State, Observation]:
        validation = self._validator.validate(self._schema, state, action)
        if not validation.ok:
            state = self._tick(state)
            return (
                state,
                Observation(
                    result="invalid action",
                    score=0.0,
                    protocol_ok=False,
                    issues=validation.issues,
                ),
            )

        if action.action_type == DispatchAction.DISPATCH:
            self._apply_dispatch(state, action)
        elif action.action_type == DispatchAction.CANCEL:
            self._apply_cancel(state, action)

        state = self._tick(state)

        # Simple shaping: valid action gets a strong score, resolving gets max.
        score = 0.8
        if any(i.status == IncidentStatus.RESOLVED for i in state.incidents.values()):
            score = 1.0

        return (
            state,
            Observation(
                result="ok",
                score=score,
                protocol_ok=True,
                issues=[],
            ),
        )

    def is_terminal(self, state: State) -> bool:
        max_steps = int(state.metadata.get("max_steps", MAX_STEPS))
        if state.step_count >= max_steps:
            return True
        if any(i.status == IncidentStatus.ESCALATED for i in state.incidents.values()):
            return True
        if state.incidents and all(
            i.status == IncidentStatus.RESOLVED for i in state.incidents.values()
        ):
            return True
        return False

    def _create_incident(self, state: State) -> IncidentState:
        self._incident_counter += 1
        incident_id = f"INC-{self._incident_counter:04d}"

        incident_type = self._rng.choice(list(IncidentType))
        severity = self._rng.choice(list(IncidentSeverity))
        width, height = self._schema.grid_size
        location_x = float(self._rng.uniform(0.0, float(width)))
        location_y = float(self._rng.uniform(0.0, float(height)))

        return IncidentState(
            incident_id=incident_id,
            incident_type=incident_type,
            severity=severity,
            location_x=location_x,
            location_y=location_y,
            reported_at_step=state.step_count,
            units_assigned=[],
            status=IncidentStatus.PENDING,
            survival_clock=_severity_deadline_seconds(severity),
        )

    def _apply_dispatch(self, state: State, action: Action) -> None:
        unit = state.units[action.unit_id]
        incident = state.incidents[action.incident_id]

        speed = float(self._schema.unit_speeds.get(unit.unit_type, 1.0))
        dist = _distance(
            unit.location_x,
            unit.location_y,
            incident.location_x,
            incident.location_y,
        )
        eta = dist / max(speed, 1e-6)

        unit.status = UnitStatus.DISPATCHED
        unit.assigned_incident_id = incident.incident_id
        unit.eta_seconds = max(0.0, float(eta))

        if unit.unit_id not in incident.units_assigned:
            incident.units_assigned.append(unit.unit_id)
        if incident.status == IncidentStatus.PENDING:
            incident.status = IncidentStatus.RESPONDING

    def _apply_cancel(self, state: State, action: Action) -> None:
        unit = state.units[action.unit_id]
        incident = state.incidents[action.incident_id]

        unit.status = UnitStatus.AVAILABLE
        unit.assigned_incident_id = None
        unit.eta_seconds = 0.0

        if unit.unit_id in incident.units_assigned:
            incident.units_assigned.remove(unit.unit_id)
        if not incident.units_assigned and incident.status in {
            IncidentStatus.RESPONDING,
            IncidentStatus.ON_SCENE,
        }:
            incident.status = IncidentStatus.PENDING
            incident.survival_clock = _severity_deadline_seconds(incident.severity)

    def _tick(self, state: State) -> State:
        state.step_count += 1
        state.city_time += DEFAULT_DT_S

        # Apply any scheduled unit status changes.
        for change in list(state.metadata.get("unit_status_changes", [])):
            if int(change.get("at_step", -1)) != state.step_count:
                continue
            unit_id = str(change.get("unit_id", ""))
            if unit_id in state.units:
                new_status = UnitStatus(change.get("status"))
                unit = state.units[unit_id]
                unit.status = new_status
                if new_status in {UnitStatus.OUT_OF_SERVICE, UnitStatus.AVAILABLE}:
                    unit.assigned_incident_id = None
                    unit.eta_seconds = 0.0

        # Spawn incident waves.
        for wave in list(state.metadata.get("waves", [])):
            if int(wave.get("at_step", -1)) != state.step_count:
                continue
            for inc in wave.get("incidents", []):
                incident_obj = IncidentState.model_validate(inc)
                if incident_obj.incident_id not in state.incidents:
                    state.incidents[incident_obj.incident_id] = incident_obj
                if (
                    incident_obj.severity == IncidentSeverity.PRIORITY_1
                    and incident_obj.incident_id not in state.metadata.get("p1_seen", [])
                ):
                    state.metadata.setdefault("p1_seen", []).append(incident_obj.incident_id)

        for unit in state.units.values():
            if unit.status == UnitStatus.DISPATCHED:
                unit.eta_seconds = max(0.0, unit.eta_seconds - DEFAULT_DT_S)
                if unit.eta_seconds <= 0.0 and unit.assigned_incident_id is not None:
                    unit.status = UnitStatus.ON_SCENE

        for incident in state.incidents.values():
            if incident.status in {IncidentStatus.PENDING, IncidentStatus.RESPONDING}:
                incident.survival_clock = max(0.0, incident.survival_clock - DEFAULT_DT_S)
                if incident.survival_clock <= 0.0:
                    incident.status = IncidentStatus.ESCALATED
                    failed = state.metadata.setdefault("failed_incidents", [])
                    if incident.incident_id not in failed:
                        failed.append(incident.incident_id)
                    continue

            if incident.status == IncidentStatus.RESPONDING:
                if any(
                    state.units[uid].status == UnitStatus.ON_SCENE
                    for uid in incident.units_assigned
                    if uid in state.units
                ):
                    incident.status = IncidentStatus.ON_SCENE
                    incident.survival_clock = _resolve_timer_seconds(incident.severity)

            if incident.status == IncidentStatus.ON_SCENE:
                incident.survival_clock = max(0.0, incident.survival_clock - DEFAULT_DT_S)
                if incident.survival_clock <= 0.0:
                    incident.status = IncidentStatus.RESOLVED
                    resolved = state.metadata.setdefault("resolved_incidents", [])
                    if incident.incident_id not in resolved:
                        resolved.append(incident.incident_id)

        return state
