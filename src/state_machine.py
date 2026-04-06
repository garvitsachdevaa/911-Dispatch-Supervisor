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
from src.rewards import RewardCalculator
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

    Supports dispatch operations and advances incidents through:
    PENDING → RESPONDING → ON_SCENE → RESOLVED.
    """

    def __init__(self, schema: CitySchema, seed: int | None = None) -> None:
        self._schema = schema
        self._seed = seed
        self._rng = random.Random(seed)
        self._validator = DispatchProtocolValidator()
        self._rewards = RewardCalculator()
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
        # Convert unit type values to plain strings for consistent lookup
        raw_required = schema_dump.get("default_required_units", {})
        converted_required: dict[str, list[str]] = {}
        for inc_type, unit_types in raw_required.items():
            inc_key = getattr(inc_type, "value", None) or str(inc_type)
            converted_required[str(inc_key)] = [
                str(getattr(u, "value", None) or str(u)) for u in list(unit_types)
            ]
        state.metadata.setdefault("default_required_units", converted_required)

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
            i for i in state.incidents.values() if i.status not in {IncidentStatus.RESOLVED}
        ]
        if not active_incidents:
            return actions

        # Keep ordering stable and DISPATCH-first for callers that take legal[0].
        active_incidents_sorted = sorted(active_incidents, key=lambda i: i.incident_id)
        units_sorted = sorted(state.units.values(), key=lambda u: u.unit_id)

        # Pick a deterministic "reference" unit for actions that don't semantically need one
        # (UPGRADE/DOWNGRADE require unit_id in the Action contract).
        ref_unit_id = units_sorted[0].unit_id if units_sorted else ""

        # DISPATCH actions (primary control surface)
        for unit in units_sorted:
            if unit.status != UnitStatus.AVAILABLE:
                continue
            for incident in active_incidents_sorted:
                actions.append(
                    Action(
                        action_type=DispatchAction.DISPATCH,
                        unit_id=unit.unit_id,
                        incident_id=incident.incident_id,
                    )
                )

        # STAGE actions (pre-position without committing as assigned)
        for unit in units_sorted:
            if unit.status != UnitStatus.AVAILABLE:
                continue
            for incident in active_incidents_sorted:
                if incident.status != IncidentStatus.PENDING:
                    continue
                actions.append(
                    Action(
                        action_type=DispatchAction.STAGE,
                        unit_id=unit.unit_id,
                        incident_id=incident.incident_id,
                    )
                )

        # CANCEL actions (release currently assigned units)
        for unit in units_sorted:
            if unit.assigned_incident_id is None:
                continue
            actions.append(
                Action(
                    action_type=DispatchAction.CANCEL,
                    unit_id=unit.unit_id,
                    incident_id=unit.assigned_incident_id,
                )
            )

        # REASSIGN actions (redirect already-assigned units to a different active incident)
        for unit in units_sorted:
            if unit.assigned_incident_id is None:
                continue
            if unit.status not in {UnitStatus.DISPATCHED, UnitStatus.ON_SCENE, UnitStatus.TRANSPORTING}:
                continue
            for incident in active_incidents_sorted:
                if incident.incident_id == unit.assigned_incident_id:
                    continue
                actions.append(
                    Action(
                        action_type=DispatchAction.REASSIGN,
                        unit_id=unit.unit_id,
                        incident_id=incident.incident_id,
                    )
                )

        # MUTUAL_AID actions (only for unit types with no local availability)
        # Use any existing unit as the "type selector".
        available_types = {u.unit_type for u in units_sorted if u.status == UnitStatus.AVAILABLE}
        type_to_template_unit: dict[object, str] = {}
        for unit in units_sorted:
            type_to_template_unit.setdefault(unit.unit_type, unit.unit_id)

        for unit_type, template_unit_id in sorted(type_to_template_unit.items(), key=lambda kv: str(kv[0])):
            if unit_type in available_types:
                continue
            for incident in active_incidents_sorted:
                actions.append(
                    Action(
                        action_type=DispatchAction.MUTUAL_AID,
                        unit_id=template_unit_id,
                        incident_id=incident.incident_id,
                    )
                )

        # UPGRADE / DOWNGRADE actions (severity adjustments)
        if ref_unit_id:
            for incident in active_incidents_sorted:
                if incident.status == IncidentStatus.RESOLVED:
                    continue

                # These candidates are filtered by protocol validation at step-time,
                # but we only generate the obviously-relevant ones.
                if incident.severity == IncidentSeverity.PRIORITY_1:
                    actions.append(
                        Action(
                            action_type=DispatchAction.DOWNGRADE,
                            unit_id=ref_unit_id,
                            incident_id=incident.incident_id,
                            priority_override=IncidentSeverity.PRIORITY_2,
                        )
                    )
                    actions.append(
                        Action(
                            action_type=DispatchAction.DOWNGRADE,
                            unit_id=ref_unit_id,
                            incident_id=incident.incident_id,
                            priority_override=IncidentSeverity.PRIORITY_3,
                        )
                    )
                elif incident.severity == IncidentSeverity.PRIORITY_2:
                    actions.append(
                        Action(
                            action_type=DispatchAction.UPGRADE,
                            unit_id=ref_unit_id,
                            incident_id=incident.incident_id,
                            priority_override=IncidentSeverity.PRIORITY_1,
                        )
                    )
                    actions.append(
                        Action(
                            action_type=DispatchAction.DOWNGRADE,
                            unit_id=ref_unit_id,
                            incident_id=incident.incident_id,
                            priority_override=IncidentSeverity.PRIORITY_3,
                        )
                    )
                else:
                    actions.append(
                        Action(
                            action_type=DispatchAction.UPGRADE,
                            unit_id=ref_unit_id,
                            incident_id=incident.incident_id,
                            priority_override=IncidentSeverity.PRIORITY_2,
                        )
                    )
                    actions.append(
                        Action(
                            action_type=DispatchAction.UPGRADE,
                            unit_id=ref_unit_id,
                            incident_id=incident.incident_id,
                            priority_override=IncidentSeverity.PRIORITY_1,
                        )
                    )

        # Filter out any actions that violate the protocol validator.
        legal: list[Action] = []
        for a in actions:
            result = self._validator.validate(self._schema, state, a)
            if result.ok:
                legal.append(a)
        return legal

    def step(self, state: State, action: Action) -> tuple[State, Observation]:
        validation = self._validator.validate(self._schema, state, action)
        if not validation.ok:
            state = self._tick(state)
            breakdown = {
                "response_time": 0.0,
                "triage": 0.0,
                "survival": 0.0,
                "coverage": 0.0,
                "protocol": 0.0,
            }
            return (
                state,
                Observation(
                    result="invalid action",
                    score=0.0,
                    protocol_ok=False,
                    issues=validation.issues,
                    reward_breakdown=breakdown,
                ),
            )

        if action.action_type == DispatchAction.DISPATCH:
            self._apply_dispatch(state, action)
        elif action.action_type == DispatchAction.CANCEL:
            self._apply_cancel(state, action)
        elif action.action_type == DispatchAction.REASSIGN:
            self._apply_reassign(state, action)
        elif action.action_type == DispatchAction.STAGE:
            self._apply_stage(state, action)
        elif action.action_type == DispatchAction.MUTUAL_AID:
            self._apply_mutual_aid(state, action)
        elif action.action_type in {DispatchAction.UPGRADE, DispatchAction.DOWNGRADE}:
            self._apply_severity_change(state, action)

        state = self._tick(state)

        obs = Observation(
            result="ok",
            score=0.0,
            protocol_ok=True,
            issues=validation.issues,
        )
        signal, total = self._rewards.compute_reward(state, action, obs)
        obs = obs.model_copy(update={"score": total, "reward_breakdown": signal.model_dump()})

        return (state, obs)

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
        # Use Manhattan distance to match move_unit_toward physics
        dx = abs(unit.location_x - incident.location_x)
        dy = abs(unit.location_y - incident.location_y)
        manhattan_dist = dx + dy
        eta = manhattan_dist / max(speed, 1e-6)

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

    def _apply_reassign(self, state: State, action: Action) -> None:
        unit = state.units[action.unit_id]
        new_incident = state.incidents[action.incident_id]

        old_incident_id = unit.assigned_incident_id
        old_incident = state.incidents.get(old_incident_id) if old_incident_id else None

        # Remove from the old incident, if present.
        if old_incident is not None and unit.unit_id in old_incident.units_assigned:
            old_incident.units_assigned.remove(unit.unit_id)
            if not old_incident.units_assigned and old_incident.status in {
                IncidentStatus.RESPONDING,
                IncidentStatus.ON_SCENE,
            }:
                old_incident.status = IncidentStatus.PENDING
                old_incident.survival_clock = _severity_deadline_seconds(old_incident.severity)

        # Assign to the new incident like a dispatch.
        unit.status = UnitStatus.DISPATCHED
        unit.assigned_incident_id = new_incident.incident_id

        speed = float(self._schema.unit_speeds.get(unit.unit_type, 1.0))
        dx = abs(unit.location_x - new_incident.location_x)
        dy = abs(unit.location_y - new_incident.location_y)
        manhattan_dist = dx + dy
        eta = manhattan_dist / max(speed, 1e-6)
        unit.eta_seconds = max(0.0, float(eta))

        if unit.unit_id not in new_incident.units_assigned:
            new_incident.units_assigned.append(unit.unit_id)
        if new_incident.status == IncidentStatus.PENDING:
            new_incident.status = IncidentStatus.RESPONDING

    def _apply_stage(self, state: State, action: Action) -> None:
        """Pre-position a unit towards an incident without counting as 'assigned'."""
        unit = state.units[action.unit_id]
        incident = state.incidents[action.incident_id]

        speed = float(self._schema.unit_speeds.get(unit.unit_type, 1.0))
        dx = abs(unit.location_x - incident.location_x)
        dy = abs(unit.location_y - incident.location_y)
        manhattan_dist = dx + dy
        eta = manhattan_dist / max(speed, 1e-6)

        unit.status = UnitStatus.DISPATCHED
        unit.assigned_incident_id = incident.incident_id
        unit.eta_seconds = max(0.0, float(eta))

    def _apply_mutual_aid(self, state: State, action: Action) -> None:
        """Request an external unit of the given type and dispatch it."""
        template = state.units[action.unit_id]
        incident = state.incidents[action.incident_id]

        counter = int(state.metadata.get("mutual_aid_counter", 0)) + 1
        state.metadata["mutual_aid_counter"] = counter

        prefix = template.unit_type.value[:3]
        new_unit_id = f"MA-{prefix}-{counter}"
        new_unit_id = new_unit_id[:20]

        speed = float(self._schema.unit_speeds.get(template.unit_type, 1.0))
        dx = abs(template.location_x - incident.location_x)
        dy = abs(template.location_y - incident.location_y)
        manhattan_dist = dx + dy
        base_eta = manhattan_dist / max(speed, 1e-6)
        penalty = float(state.metadata.get("mutual_aid_eta_penalty", 120.0))

        unit = UnitState(
            unit_id=new_unit_id,
            unit_type=template.unit_type,
            status=UnitStatus.DISPATCHED,
            location_x=float(template.location_x),
            location_y=float(template.location_y),
            assigned_incident_id=incident.incident_id,
            eta_seconds=max(0.0, float(base_eta + penalty)),
            crew_count=int(template.crew_count),
        )
        state.units[unit.unit_id] = unit

        if unit.unit_id not in incident.units_assigned:
            incident.units_assigned.append(unit.unit_id)
        if incident.status == IncidentStatus.PENDING:
            incident.status = IncidentStatus.RESPONDING

    def _apply_severity_change(self, state: State, action: Action) -> None:
        if action.priority_override is None:
            return
        incident = state.incidents[action.incident_id]
        incident.severity = action.priority_override

        # Update clocks based on current incident phase.
        if incident.status in {IncidentStatus.PENDING, IncidentStatus.RESPONDING}:
            incident.survival_clock = _severity_deadline_seconds(incident.severity)
        elif incident.status == IncidentStatus.ON_SCENE:
            incident.survival_clock = _resolve_timer_seconds(incident.severity)

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
