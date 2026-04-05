"""OpenEnv-compatible environment wrapper."""

import uuid

from src.city_schema import CitySchemaLoader
from src.models import Action, Observation, State
from src.state_machine import DispatchStateMachine


class OpenEnvEnvironment:
    def __init__(self, task_id: str, seed: int | None = None) -> None:
        self.task_id = task_id
        self.seed_value = seed
        schema = CitySchemaLoader.load("metro_city")
        self._machine = DispatchStateMachine(schema=schema, seed=seed)
        self._state: State | None = None
        self._last_observation: Observation | None = None

    async def reset(self) -> Observation:
        episode_id = str(uuid.uuid4())
        self._state = self._machine.reset(task_id=self.task_id, episode_id=episode_id)
        self._last_observation = Observation(
            result="dispatch center online",
            score=0.0,
            protocol_ok=True,
            issues=[],
        )
        return self._last_observation

    async def step(self, action: Action) -> tuple[Observation, float, bool]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        state, obs = self._machine.step(self._state, action)
        self._state = state
        self._last_observation = obs
        done = self._machine.is_terminal(state)
        return obs, obs.score, done

    def state(self) -> State:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def last_observation(self) -> Observation | None:
        return self._last_observation

    def legal_actions(self) -> list[Action]:
        if self._state is None:
            return []
        return self._machine.get_legal_actions(self._state)

    def close(self) -> None:
        self._state = None
        self._last_observation = None
