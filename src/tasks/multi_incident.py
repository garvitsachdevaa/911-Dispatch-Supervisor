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
        """Grade based on: P1 incidents resolved, triage correctness, coverage."""
        if not rewards:
            return 0.0

        total = len(state.incidents)
        if total == 0:
            return 0.0

        resolved = sum(1 for i in state.incidents.values() if i.status.value == "RESOLVED")
        failed = sum(1 for i in state.incidents.values() if i.status.value == "ESCALATED")
        p1_total = sum(1 for i in state.incidents.values() if i.severity.value == "PRIORITY_1")
        p1_resolved = sum(
            1
            for iid in state.metadata.get("resolved_incidents", [])
            if state.incidents.get(iid)
            and state.incidents[iid].severity.value == "PRIORITY_1"
        )

        resolution_score = resolved / total
        p1_score = (p1_resolved / p1_total) if p1_total > 0 else 1.0
        failure_penalty = failed / total

        score = 0.5 * p1_score + 0.3 * resolution_score - 0.2 * failure_penalty
        return max(0.0, min(1.0, score))
