"""Task registry and deterministic scenario fixtures for the 911 dispatch environment."""

from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models import (
    IncidentSeverity,
    IncidentStatus,
    IncidentType,
    UnitState,
    UnitStatus,
    UnitType,
)


class TaskInfo(BaseModel):
    """Task information metadata."""

    model_config = {"extra": "forbid"}

    task_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    difficulty: Literal["easy", "medium", "hard"]
    initial_state_fn: str = Field(..., min_length=1)


class TaskRegistry:
    """Registry for managing available tasks."""

    REGISTRY: dict[str, TaskInfo] = {}

    @classmethod
    def register(cls, task: TaskInfo) -> None:
        cls.REGISTRY[task.task_id] = task

    @classmethod
    def get(cls, task_id: str) -> TaskInfo:
        if task_id not in cls.REGISTRY:
            raise KeyError(f"Task '{task_id}' not found in registry")
        return cls.REGISTRY[task_id]

    @classmethod
    def list_tasks(cls) -> list[TaskInfo]:
        return list(cls.REGISTRY.values())


TaskRegistry.register(
    TaskInfo(
        task_id="single_incident",
        name="Single Incident Response",
        description="1 incident, right unit, fast dispatch",
        difficulty="easy",
        initial_state_fn="build_single_incident_fixture",
    )
)

TaskRegistry.register(
    TaskInfo(
        task_id="multi_incident",
        name="Simultaneous Multi-Incident",
        description="3 concurrent incidents requiring triage",
        difficulty="medium",
        initial_state_fn="build_multi_incident_fixture",
    )
)

TaskRegistry.register(
    TaskInfo(
        task_id="mass_casualty",
        name="Mass Casualty Event",
        description="wave-based incidents with resource conflict",
        difficulty="hard",
        initial_state_fn="build_mass_casualty_fixture",
    )
)

TaskRegistry.register(
    TaskInfo(
        task_id="shift_surge",
        name="Shift Surge",
        description="units go out of service + steady incident stream",
        difficulty="hard",
        initial_state_fn="build_shift_surge_fixture",
    )
)


class DispatchScenarioFactory:
    """Factory for creating deterministic dispatch scenario fixtures.

    Returns `(state_dict, meta_dict)`.
    """

    @staticmethod
    def _seeded_random(seed: int) -> random.Random:
        return random.Random(seed)

    @classmethod
    def build(cls, task_id: str, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        task = TaskRegistry.get(task_id)
        fn_name = task.initial_state_fn

        if fn_name == "build_single_incident_fixture":
            return cls.build_single_incident_fixture(seed)
        if fn_name == "build_multi_incident_fixture":
            return cls.build_multi_incident_fixture(seed)
        if fn_name == "build_mass_casualty_fixture":
            return cls.build_mass_casualty_fixture(seed)
        if fn_name == "build_shift_surge_fixture":
            return cls.build_shift_surge_fixture(seed)

        raise ValueError(f"Unknown initial_state_fn: {fn_name}")

    @classmethod
    def _base_units_city_small(cls) -> dict[str, UnitState]:
        return {
            "MED-1": UnitState(
                unit_id="MED-1",
                unit_type=UnitType.MEDIC,
                status=UnitStatus.AVAILABLE,
                location_x=10.0,
                location_y=10.0,
                assigned_incident_id=None,
                eta_seconds=0.0,
                crew_count=2,
            ),
            "ENG-1": UnitState(
                unit_id="ENG-1",
                unit_type=UnitType.ENGINE,
                status=UnitStatus.AVAILABLE,
                location_x=20.0,
                location_y=20.0,
                assigned_incident_id=None,
                eta_seconds=0.0,
                crew_count=4,
            ),
            "PAT-1": UnitState(
                unit_id="PAT-1",
                unit_type=UnitType.PATROL,
                status=UnitStatus.AVAILABLE,
                location_x=30.0,
                location_y=30.0,
                assigned_incident_id=None,
                eta_seconds=0.0,
                crew_count=2,
            ),
        }

    @classmethod
    def build_single_incident_fixture(cls, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = cls._seeded_random(seed)
        units = cls._base_units_city_small()

        incidents = {
            "INC-001": {
                "incident_id": "INC-001",
                "incident_type": IncidentType.CARDIAC_ARREST,
                "severity": IncidentSeverity.PRIORITY_1,
                "location_x": 12.0 + rng.uniform(-1.0, 1.0),
                "location_y": 12.0 + rng.uniform(-1.0, 1.0),
                "reported_at_step": 0,
                "units_assigned": [],
                "status": IncidentStatus.PENDING,
                "survival_clock": 600.0,
            }
        }

        state_dict = {
            "units": {k: v.model_dump() for k, v in units.items()},
            "incidents": incidents,
            "episode_id": f"single-{seed}",
            "step_count": 0,
            "task_id": "single_incident",
            "city_time": 0.0,
            "metadata": {},
        }

        meta = {
            "max_steps": 20,
            "waves": [],
            "districts": ["downtown", "northside", "eastport", "suburbs", "industrial"],
            "grid_size": [100, 100],
        }
        return state_dict, meta

    @classmethod
    def build_multi_incident_fixture(cls, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = cls._seeded_random(seed)

        units: dict[str, UnitState] = {
            **cls._base_units_city_small(),
            "MED-2": UnitState(
                unit_id="MED-2",
                unit_type=UnitType.MEDIC,
                status=UnitStatus.AVAILABLE,
                location_x=70.0,
                location_y=30.0,
                assigned_incident_id=None,
                eta_seconds=0.0,
                crew_count=2,
            ),
            "LAD-1": UnitState(
                unit_id="LAD-1",
                unit_type=UnitType.LADDER,
                status=UnitStatus.AVAILABLE,
                location_x=10.0,
                location_y=20.0,
                assigned_incident_id=None,
                eta_seconds=0.0,
                crew_count=5,
            ),
            "ENG-2": UnitState(
                unit_id="ENG-2",
                unit_type=UnitType.ENGINE,
                status=UnitStatus.AVAILABLE,
                location_x=50.0,
                location_y=50.0,
                assigned_incident_id=None,
                eta_seconds=0.0,
                crew_count=4,
            ),
        }

        incidents = {
            "INC-001": {
                "incident_id": "INC-001",
                "incident_type": IncidentType.STRUCTURE_FIRE,
                "severity": IncidentSeverity.PRIORITY_2,
                "location_x": 20.0 + rng.uniform(-2.0, 2.0),
                "location_y": 80.0 + rng.uniform(-2.0, 2.0),
                "reported_at_step": 0,
                "units_assigned": [],
                "status": IncidentStatus.PENDING,
                "survival_clock": 1200.0,
            },
            "INC-002": {
                "incident_id": "INC-002",
                "incident_type": IncidentType.CARDIAC_ARREST,
                "severity": IncidentSeverity.PRIORITY_1,
                "location_x": 14.0 + rng.uniform(-2.0, 2.0),
                "location_y": 22.0 + rng.uniform(-2.0, 2.0),
                "reported_at_step": 0,
                "units_assigned": [],
                "status": IncidentStatus.PENDING,
                "survival_clock": 600.0,
            },
            "INC-003": {
                "incident_id": "INC-003",
                "incident_type": IncidentType.SHOOTING,
                "severity": IncidentSeverity.PRIORITY_1,
                "location_x": 75.0 + rng.uniform(-2.0, 2.0),
                "location_y": 15.0 + rng.uniform(-2.0, 2.0),
                "reported_at_step": 0,
                "units_assigned": [],
                "status": IncidentStatus.PENDING,
                "survival_clock": 600.0,
            },
        }

        state_dict = {
            "units": {k: v.model_dump() for k, v in units.items()},
            "incidents": incidents,
            "episode_id": f"multi-{seed}",
            "step_count": 0,
            "task_id": "multi_incident",
            "city_time": 0.0,
            "metadata": {},
        }

        meta = {
            "max_steps": 40,
            "waves": [],
            "districts": ["downtown", "northside", "eastport", "suburbs", "industrial"],
            "grid_size": [100, 100],
        }
        return state_dict, meta

    @classmethod
    def build_mass_casualty_fixture(cls, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = cls._seeded_random(seed)

        units: dict[str, UnitState] = {
            "ENG-1": UnitState(unit_id="ENG-1", unit_type=UnitType.ENGINE, status=UnitStatus.AVAILABLE, location_x=10.0, location_y=20.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=4),
            "ENG-2": UnitState(unit_id="ENG-2", unit_type=UnitType.ENGINE, status=UnitStatus.AVAILABLE, location_x=50.0, location_y=50.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=4),
            "MED-1": UnitState(unit_id="MED-1", unit_type=UnitType.MEDIC, status=UnitStatus.AVAILABLE, location_x=15.0, location_y=25.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
            "MED-2": UnitState(unit_id="MED-2", unit_type=UnitType.MEDIC, status=UnitStatus.AVAILABLE, location_x=70.0, location_y=30.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
            "LAD-1": UnitState(unit_id="LAD-1", unit_type=UnitType.LADDER, status=UnitStatus.AVAILABLE, location_x=10.0, location_y=20.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=5),
            "PAT-1": UnitState(unit_id="PAT-1", unit_type=UnitType.PATROL, status=UnitStatus.AVAILABLE, location_x=30.0, location_y=60.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
            "PAT-2": UnitState(unit_id="PAT-2", unit_type=UnitType.PATROL, status=UnitStatus.AVAILABLE, location_x=80.0, location_y=10.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
        }

        incidents = {
            "INC-001": {
                "incident_id": "INC-001",
                "incident_type": IncidentType.BUILDING_COLLAPSE,
                "severity": IncidentSeverity.PRIORITY_1,
                "location_x": 45.0 + rng.uniform(-3.0, 3.0),
                "location_y": 45.0 + rng.uniform(-3.0, 3.0),
                "reported_at_step": 0,
                "units_assigned": [],
                "status": IncidentStatus.PENDING,
                "survival_clock": 600.0,
            }
        }

        waves = [
            {
                "at_step": 5,
                "incidents": [
                    {
                        "incident_id": "INC-002",
                        "incident_type": IncidentType.STRUCTURE_FIRE,
                        "severity": IncidentSeverity.PRIORITY_2,
                        "location_x": 10.0 + rng.uniform(-3.0, 3.0),
                        "location_y": 90.0 + rng.uniform(-3.0, 3.0),
                        "reported_at_step": 5,
                        "units_assigned": [],
                        "status": IncidentStatus.PENDING,
                        "survival_clock": 1200.0,
                    }
                ],
            },
            {
                "at_step": 12,
                "incidents": [
                    {
                        "incident_id": "INC-003",
                        "incident_type": IncidentType.CARDIAC_ARREST,
                        "severity": IncidentSeverity.PRIORITY_1,
                        "location_x": 85.0 + rng.uniform(-3.0, 3.0),
                        "location_y": 15.0 + rng.uniform(-3.0, 3.0),
                        "reported_at_step": 12,
                        "units_assigned": [],
                        "status": IncidentStatus.PENDING,
                        "survival_clock": 600.0,
                    },
                    {
                        "incident_id": "INC-004",
                        "incident_type": IncidentType.CARDIAC_ARREST,
                        "severity": IncidentSeverity.PRIORITY_1,
                        "location_x": 15.0 + rng.uniform(-3.0, 3.0),
                        "location_y": 10.0 + rng.uniform(-3.0, 3.0),
                        "reported_at_step": 12,
                        "units_assigned": [],
                        "status": IncidentStatus.PENDING,
                        "survival_clock": 600.0,
                    },
                ],
            },
        ]

        state_dict = {
            "units": {k: v.model_dump() for k, v in units.items()},
            "incidents": incidents,
            "episode_id": f"mass-{seed}",
            "step_count": 0,
            "task_id": "mass_casualty",
            "city_time": 0.0,
            "metadata": {},
        }

        meta = {
            "max_steps": 60,
            "waves": waves,
            "mutual_aid_eta_penalty": 120.0,
            "districts": ["downtown", "northside", "eastport", "suburbs", "industrial"],
            "grid_size": [100, 100],
        }
        return state_dict, meta

    @classmethod
    def build_shift_surge_fixture(cls, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        rng = cls._seeded_random(seed)

        units: dict[str, UnitState] = {
            "ENG-1": UnitState(unit_id="ENG-1", unit_type=UnitType.ENGINE, status=UnitStatus.AVAILABLE, location_x=10.0, location_y=20.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=4),
            "MED-1": UnitState(unit_id="MED-1", unit_type=UnitType.MEDIC, status=UnitStatus.AVAILABLE, location_x=15.0, location_y=25.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
            "PAT-1": UnitState(unit_id="PAT-1", unit_type=UnitType.PATROL, status=UnitStatus.AVAILABLE, location_x=30.0, location_y=60.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
            "PAT-2": UnitState(unit_id="PAT-2", unit_type=UnitType.PATROL, status=UnitStatus.AVAILABLE, location_x=80.0, location_y=10.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=2),
            "ENG-2": UnitState(unit_id="ENG-2", unit_type=UnitType.ENGINE, status=UnitStatus.AVAILABLE, location_x=50.0, location_y=50.0, assigned_incident_id=None, eta_seconds=0.0, crew_count=4),
        }

        waves: list[dict[str, Any]] = []
        next_id = 0
        for t in range(0, 56, 8):
            next_id += 1
            waves.append(
                {
                    "at_step": t,
                    "incidents": [
                        {
                            "incident_id": f"INC-{next_id:03d}",
                            "incident_type": rng.choice(list(IncidentType)),
                            "severity": rng.choice(list(IncidentSeverity)),
                            "location_x": rng.uniform(0.0, 100.0),
                            "location_y": rng.uniform(0.0, 100.0),
                            "reported_at_step": t,
                            "units_assigned": [],
                            "status": IncidentStatus.PENDING,
                            "survival_clock": 900.0,
                        }
                    ],
                }
            )

        unit_status_changes = [
            {"at_step": 1, "unit_id": "PAT-2", "status": UnitStatus.OUT_OF_SERVICE},
            {"at_step": 3, "unit_id": "ENG-2", "status": UnitStatus.OUT_OF_SERVICE},
            {"at_step": 5, "unit_id": "PAT-1", "status": UnitStatus.OUT_OF_SERVICE},
        ]

        state_dict = {
            "units": {k: v.model_dump() for k, v in units.items()},
            "incidents": {},
            "episode_id": f"surge-{seed}",
            "step_count": 0,
            "task_id": "shift_surge",
            "city_time": 0.0,
            "metadata": {},
        }

        meta = {
            "max_steps": 60,
            "waves": waves,
            "unit_status_changes": unit_status_changes,
            "districts": ["downtown", "northside", "eastport", "suburbs", "industrial"],
            "grid_size": [100, 100],
        }
        return state_dict, meta
