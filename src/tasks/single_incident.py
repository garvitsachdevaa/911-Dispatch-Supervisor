"""Task: Single Incident Response (Easy)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.city_schema import CitySchema
from src.models import Action, DispatchAction, IncidentSeverity, IncidentStatus, IncidentType, State, UnitType
from src.rewards import RewardCalculator
from src.state_machine import DispatchStateMachine


class SingleIncidentTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    city_schema: CitySchema
    seed: int | None = None
    state_machine: DispatchStateMachine = Field(default=None, exclude=True)

    def __init__(self, **data) -> None:
        super().__init__(**data)
        object.__setattr__(
            self,
            "state_machine",
            DispatchStateMachine(schema=self.city_schema, seed=self.seed),
        )

    def reset(self, episode_id: str) -> State:
        return self.state_machine.reset(task_id="single_incident", episode_id=episode_id)

    def step(self, state: State, action: Action) -> tuple[State, object]:
        return self.state_machine.step(state, action)

    def is_terminal(self, state: State) -> bool:
        return self.state_machine.is_terminal(state)


class SingleIncidentGrader:
    def __init__(self) -> None:
        self.reward_calculator = RewardCalculator()

    def grade(self, state: State, rewards: list[float]) -> float:
        """Grade based on: correct unit dispatched, fast response, incident resolved."""
        if not rewards:
            return 0.0

        incident = state.incidents.get("INC-001")
        if incident is None:
            return 0.0

        score = 0.0

        # Component 1: Was the incident resolved? (50% weight)
        if incident.status.value == "RESOLVED":
            score += 0.50

        # Component 2: Correct unit type dispatched? (30% weight)
        medic_dispatched = any(
            u.unit_type.value == "MEDIC"
            and (
                u.assigned_incident_id == "INC-001"
                or u.status.value in {"ON_SCENE", "DISPATCHED"}
            )
            for u in state.units.values()
        )
        if medic_dispatched:
            score += 0.30

        # Component 3: Speed — resolved within first 10 steps (20% weight)
        if incident.status.value == "RESOLVED" and state.step_count <= 10:
            score += 0.20

        return max(0.0, min(1.0, score))
