"""Task: Shift Surge (Medium-Hard)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.city_schema import CitySchema
from src.models import Action, IncidentStatus, State, UnitStatus
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
        """Grade long-horizon surge management.

        Emphasizes:
        - Resolving incidents (throughput)
        - Preventing escalations (failures)
        - Keeping queue/backlog low (pending/responding)
        - Priority-1 survival outcomes
        - Maintaining geographic coverage
        """

        if not rewards:
            return 0.0

        total_incidents = len(state.incidents)
        if total_incidents == 0:
            return 0.0

        resolved = sum(1 for i in state.incidents.values() if i.status == IncidentStatus.RESOLVED)
        failed = sum(1 for i in state.incidents.values() if i.status == IncidentStatus.ESCALATED)
        backlog = sum(
            1
            for i in state.incidents.values()
            if i.status in {IncidentStatus.PENDING, IncidentStatus.RESPONDING}
        )

        resolved_ratio = resolved / total_incidents
        failed_ratio = failed / total_incidents
        backlog_ratio = backlog / total_incidents

        p1_survival = float(self.reward_calculator._compute_survival(state))
        coverage = float(self.reward_calculator._compute_coverage(state))
        mean_reward = float(sum(rewards) / max(len(rewards), 1))

        score = (
            0.35 * resolved_ratio
            + 0.25 * p1_survival
            + 0.15 * coverage
            + 0.15 * max(0.0, 1.0 - backlog_ratio)
            + 0.10 * max(0.0, min(1.0, mean_reward))
            - 0.25 * failed_ratio
        )

        return max(0.0, min(1.0, float(score)))
