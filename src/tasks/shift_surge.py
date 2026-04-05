"""Task: Shift Surge (Medium-Hard)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.city_schema import CitySchema
from src.models import Action, State, UnitStatus
from src.rewards import RewardCalculator
from src.state_machine import DispatchStateMachine


class ShiftSurgeTask(BaseModel):
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
        return self.state_machine.reset(task_id="shift_surge", episode_id=episode_id)

    def step(self, state: State, action: Action) -> tuple[State, object]:
        return self.state_machine.step(state, action)

    def is_terminal(self, state: State) -> bool:
        return self.state_machine.is_terminal(state)


class ShiftSurgeGrader:
    def __init__(self) -> None:
        self.reward_calculator = RewardCalculator()

    def grade(self, state: State, rewards: list[float]) -> float:
        if not rewards:
            return 0.0

        mean_reward = sum(rewards) / len(rewards)
        available = sum(1 for u in state.units.values() if u.status == UnitStatus.AVAILABLE)
        # Coverage proxy: keep at least 2 units available.
        bonus = 0.1 if available >= 2 else 0.0
        return max(0.0, min(1.0, mean_reward + bonus))
