"""Task: Mass Casualty Event (Hard)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.city_schema import CitySchema
from src.models import Action, IncidentSeverity, IncidentType, State
from src.rewards import RewardCalculator
from src.state_machine import DispatchStateMachine


class MassCasualtyTask(BaseModel):
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
        return self.state_machine.reset(task_id="mass_casualty", episode_id=episode_id)

    def step(self, state: State, action: Action) -> tuple[State, object]:
        return self.state_machine.step(state, action)

    def is_terminal(self, state: State) -> bool:
        return self.state_machine.is_terminal(state)


class MassCasualtyGrader:
    def __init__(self) -> None:
        self.reward_calculator = RewardCalculator()

    def grade(self, state: State, rewards: list[float]) -> float:
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)
        p1_total = sum(1 for i in state.incidents.values() if i.severity == IncidentSeverity.PRIORITY_1)
        resolved_p1 = sum(
            1
            for incident_id in state.metadata.get("resolved_incidents", [])
            if state.incidents.get(incident_id, None) is not None
            and state.incidents[incident_id].severity == IncidentSeverity.PRIORITY_1
        )

        survival_ratio = resolved_p1 / max(p1_total, 1)
        return max(0.0, min(1.0, 0.3 * mean_reward + 0.7 * survival_ratio))
