"""OpenEnv-compatible environment wrapper."""

import uuid

from src.city_schema import CitySchemaLoader
from src.grading import grade_episode
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
        self._state.metadata["cumulative_reward"] = 0.0
        self._state.metadata["episode_rewards"] = []
        self._state.metadata["episode_score"] = 0.0
        active_p1 = sum(
            1
            for i in self._state.incidents.values()
            if i.severity.value == "PRIORITY_1" and i.status.value not in {"RESOLVED", "ESCALATED"}
        )
        avail = sum(1 for u in self._state.units.values() if u.status.value == "AVAILABLE")

        self._last_observation = Observation(
            result="dispatch center online",
            score=0.0,
            protocol_ok=True,
            issues=[],
            reward_breakdown={
                "response_time": 0.0,
                "triage": 0.0,
                "survival": 0.0,
                "coverage": 0.0,
                "protocol": 1.0,
            },
            phraseology_score=1.0,
            active_p1_count=active_p1,
            units_available=avail,
        )
        return self._last_observation

    async def step(self, action: Action) -> tuple[Observation, float, bool]:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        state, obs = self._machine.step(self._state, action)
        self._state = state

        # `DispatchStateMachine.step()` sets `obs.score` to the per-step reward.
        # OpenEnv consumers often interpret `observation.score` as an episode score,
        # so we keep the per-step reward in `reward` and publish the episode score
        # into `observation.score`.
        step_reward = float(obs.score)

        rewards: list[float] = list(self._state.metadata.get("episode_rewards", []))
        rewards.append(step_reward)
        self._state.metadata["episode_rewards"] = rewards

        cumulative = float(self._state.metadata.get("cumulative_reward", 0.0))
        self._state.metadata["cumulative_reward"] = cumulative + step_reward

        # Episode score is derived from the same grading logic as benchmark runs.
        episode_score = grade_episode(task_id=self.task_id, state=self._state, rewards=rewards)
        episode_score = max(0.0, min(1.0, float(episode_score)))
        self._state.metadata["episode_score"] = episode_score

        done = self._machine.is_terminal(state)

        active_p1_count = sum(
            1
            for i in state.incidents.values()
            if i.severity.value == "PRIORITY_1" and i.status.value not in {"RESOLVED", "ESCALATED"}
        )
        units_available = sum(1 for u in state.units.values() if u.status.value == "AVAILABLE")
        
        phraseology = 0.0
        if obs.reward_breakdown:
            phraseology = obs.reward_breakdown.get("protocol", 0.0)

        obs = obs.model_copy(
            update={
                "score": episode_score,
                "phraseology_score": phraseology,
                "active_p1_count": active_p1_count,
                "units_available": units_available,
                "step_count": state.step_count,
                "episode_done": done,
            }
        )
        self._last_observation = obs
        return obs, step_reward, done

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
