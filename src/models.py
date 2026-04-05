"""Pydantic typed contracts for the 911 dispatch supervisor environment."""

from enum import StrEnum

from pydantic import BaseModel, Field


class IncidentType(StrEnum):
    STRUCTURE_FIRE = "STRUCTURE_FIRE"
    CARDIAC_ARREST = "CARDIAC_ARREST"
    MULTI_VEHICLE_ACCIDENT = "MULTI_VEHICLE_ACCIDENT"
    SHOOTING = "SHOOTING"
    OVERDOSE = "OVERDOSE"
    BUILDING_COLLAPSE = "BUILDING_COLLAPSE"
    HAZMAT_SPILL = "HAZMAT_SPILL"
    MISSING_PERSON = "MISSING_PERSON"


class IncidentSeverity(StrEnum):
    PRIORITY_1 = "PRIORITY_1"
    PRIORITY_2 = "PRIORITY_2"
    PRIORITY_3 = "PRIORITY_3"


class UnitType(StrEnum):
    ENGINE = "ENGINE"
    LADDER = "LADDER"
    MEDIC = "MEDIC"
    PATROL = "PATROL"
    SUPERVISOR = "SUPERVISOR"
    HAZMAT = "HAZMAT"


class UnitStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    DISPATCHED = "DISPATCHED"
    ON_SCENE = "ON_SCENE"
    TRANSPORTING = "TRANSPORTING"
    OUT_OF_SERVICE = "OUT_OF_SERVICE"


class IncidentStatus(StrEnum):
    PENDING = "PENDING"
    RESPONDING = "RESPONDING"
    ON_SCENE = "ON_SCENE"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"


class DispatchAction(StrEnum):
    DISPATCH = "DISPATCH"
    UPGRADE = "UPGRADE"
    DOWNGRADE = "DOWNGRADE"
    MUTUAL_AID = "MUTUAL_AID"
    CANCEL = "CANCEL"
    REASSIGN = "REASSIGN"
    STAGE = "STAGE"


class Action(BaseModel):
    """Structured dispatcher command."""

    model_config = {"extra": "forbid"}

    action_type: DispatchAction
    unit_id: str = Field(..., min_length=1, max_length=20)
    incident_id: str = Field(..., min_length=1, max_length=20)
    notes: str | None = None
    priority_override: IncidentSeverity | None = None


class Observation(BaseModel):
    """Environment response to a dispatch action."""

    model_config = {"extra": "forbid"}

    result: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    protocol_ok: bool = False
    issues: list[str] = Field(default_factory=list)
    reward_breakdown: dict[str, float] | None = None


class UnitState(BaseModel):
    """City-wide unit state."""

    model_config = {"extra": "forbid"}

    unit_id: str = Field(..., min_length=1, max_length=20)
    unit_type: UnitType
    status: UnitStatus
    location_x: float = Field(..., ge=0.0)
    location_y: float = Field(..., ge=0.0)
    assigned_incident_id: str | None = None
    eta_seconds: float = Field(default=0.0, ge=0.0)
    crew_count: int = Field(default=1, ge=0)


class IncidentState(BaseModel):
    """Incident state tracked by dispatch."""

    model_config = {"extra": "forbid"}

    incident_id: str = Field(..., min_length=1, max_length=20)
    incident_type: IncidentType
    severity: IncidentSeverity
    location_x: float = Field(..., ge=0.0)
    location_y: float = Field(..., ge=0.0)
    reported_at_step: int = Field(default=0, ge=0)
    units_assigned: list[str] = Field(default_factory=list)
    status: IncidentStatus
    survival_clock: float = Field(default=0.0, ge=0.0)


class State(BaseModel):
    """Full environment state."""

    model_config = {"extra": "forbid"}

    units: dict[str, UnitState] = Field(default_factory=dict)
    incidents: dict[str, IncidentState] = Field(default_factory=dict)
    episode_id: str = Field(..., min_length=1)
    step_count: int = Field(default=0, ge=0)
    task_id: str = Field(..., min_length=1)
    city_time: float = Field(default=0.0, ge=0.0)
    metadata: dict = Field(default_factory=dict)
