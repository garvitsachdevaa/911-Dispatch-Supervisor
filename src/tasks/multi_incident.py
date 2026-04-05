"""Task: Simultaneous Multi-Incident (Medium)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.city_schema import CitySchema
from src.models import Action, IncidentType, State, UnitType
from src.rewards import RewardCalculator
from src.state_machine import DispatchStateMachine


class MultiIncidentTask(BaseModel):
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
        return self.state_machine.reset(task_id="multi_incident", episode_id=episode_id)

    def step(self, state: State, action: Action) -> tuple[State, object]:
        return self.state_machine.step(state, action)

    def is_terminal(self, state: State) -> bool:
        return self.state_machine.is_terminal(state)


class MultiIncidentGrader:
    def __init__(self) -> None:
        self.reward_calculator = RewardCalculator()

    def grade(self, state: State, rewards: list[float]) -> float:
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)

        # Heuristic: reward triage if at least one MEDIC is assigned to the cardiac arrest.
        cardiac = next((i for i in state.incidents.values() if i.incident_type == IncidentType.CARDIAC_ARREST), None)
        if cardiac is None:
            return max(0.0, min(1.0, mean_reward))

        has_medic = any(
            (u.unit_type == UnitType.MEDIC and u.assigned_incident_id == cardiac.incident_id)
            for u in state.units.values()
        )

        bonus = 0.1 if has_medic else 0.0
        return max(0.0, min(1.0, mean_reward + bonus))
