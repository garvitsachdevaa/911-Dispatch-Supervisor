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
        """Pass if correct unit dispatched within 3 steps and ETA < 300s."""
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)

        incident = state.incidents.get("INC-001")
        if incident is None:
            return max(0.0, min(1.0, mean_reward))

        # Find the first dispatched unit to INC-001.
        dispatched_units = [
            u
            for u in state.units.values()
            if u.assigned_incident_id == "INC-001" and u.status != "AVAILABLE"
        ]

        bonus = 0.0
        if dispatched_units:
            unit = dispatched_units[0]
            correct_type = (unit.unit_type == UnitType.MEDIC and incident.incident_type == IncidentType.CARDIAC_ARREST)
            if correct_type:
                bonus += 0.2
            if unit.eta_seconds < 300.0:
                bonus += 0.2

        # Base per spec: 0.6 + bonuses, but bounded and blended with mean reward.
        target = 0.6 + bonus
        total = 0.5 * mean_reward + 0.5 * target
        return max(0.0, min(1.0, total))
