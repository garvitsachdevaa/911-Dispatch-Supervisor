"""Tests for OpenEnv server/client integration."""

import pytest
from fastapi.testclient import TestClient

from src.models import Action, ClearanceType, LifecyclePhase
from src.openenv_environment import OpenEnvEnvironment
from src.server import app as server_app


@pytest.fixture(autouse=True)
def reset_env():
    server_app._env = None
    yield
    server_app._env = None


class TestOpenEnvEnvironment:
    @pytest.mark.asyncio
    async def test_init_with_task_id_and_seed(self) -> None:
        env = OpenEnvEnvironment(task_id="arrival", seed=42)
        assert env.task_id == "arrival"
        assert env.seed_value == 42

    @pytest.mark.asyncio
    async def test_init_without_seed(self) -> None:
        env = OpenEnvEnvironment(task_id="departure")
        assert env.task_id == "departure"
        assert env.seed_value is None

    @pytest.mark.asyncio
    async def test_reset_returns_observation(self) -> None:
        env = OpenEnvEnvironment(task_id="arrival", seed=42)
        obs = await env.reset()
        assert obs.result == "reset complete"
        assert obs.score == 0.0
        assert obs.phraseology_ok is True
        assert obs.issues == []

    @pytest.mark.asyncio
    async def test_step_returns_tuple(self) -> None:
        env = OpenEnvEnvironment(task_id="arrival", seed=42)
        await env.reset()
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
        )
        obs, reward, done = await env.step(action)
        assert isinstance(obs.result, str)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    @pytest.mark.asyncio
    async def test_state_returns_state(self) -> None:
        env = OpenEnvEnvironment(task_id="arrival", seed=42)
        await env.reset()
        state = env.state()
        assert state.phase == LifecyclePhase.APPROACH
        assert state.task_id == "arrival"
        assert state.episode_id is not None
        assert state.step_count == 0

    @pytest.mark.asyncio
    async def test_close_cleans_up(self) -> None:
        env = OpenEnvEnvironment(task_id="arrival", seed=42)
        await env.reset()
        env.close()
        assert env._lifecycle_state is None


class TestResetEndpoint:
    def test_reset_returns_observation(self) -> None:
        c = TestClient(server_app.app)
        response = c.post("/reset", json={"task_id": "arrival", "seed": 42})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "reset complete"
        assert data["score"] == 0.0

    def test_reset_without_seed(self) -> None:
        c = TestClient(server_app.app)
        response = c.post("/reset", json={"task_id": "departure"})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "reset complete"

    def test_reset_unknown_task_still_works(self) -> None:
        c = TestClient(server_app.app)
        response = c.post("/reset", json={"task_id": "unknown_task"})
        assert response.status_code == 200


class TestStepEndpoint:
    def test_step_requires_reset_first(self) -> None:
        c = TestClient(server_app.app)
        response = c.post(
            "/step",
            json={"action": {"clearance_type": "taxi", "target_callsign": "BAW123"}},
        )
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"]

    def test_step_returns_observation_reward_done(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival"})
        response = c.post(
            "/step",
            json={"action": {"clearance_type": "landing", "target_callsign": "BAW123"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert isinstance(data["observation"], dict)
        assert isinstance(data["reward"], float)
        assert isinstance(data["done"], bool)

    def test_step_with_invalid_action_rejected(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival"})
        response = c.post(
            "/step",
            json={"action": {"invalid_field": "value"}},
        )
        assert response.status_code == 500
        assert "Invalid action" in response.json()["detail"]

    def test_step_with_illegal_action_returns_illegal_transition(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival"})
        response = c.post(
            "/step",
            json={"action": {"clearance_type": "taxi", "target_callsign": "UNKNOWN"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["observation"]["result"] == "illegal_transition"


class TestStateEndpoint:
    def test_state_requires_reset_first(self) -> None:
        c = TestClient(server_app.app)
        response = c.get("/state")
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"]

    def test_state_returns_current_state(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival", "seed": 42})
        response = c.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert data["phase"] == "approach"
        assert data["task_id"] == "arrival"
        assert data["step_count"] == 0

    def test_state_reflects_step_count(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival", "seed": 42})
        c.post(
            "/step",
            json={"action": {"clearance_type": "landing", "target_callsign": "BAW123"}},
        )
        response = c.get("/state")
        data = response.json()
        assert data["step_count"] == 1


class TestHealthEndpoint:
    def test_health_returns_ok(self) -> None:
        c = TestClient(server_app.app)
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestTaskRouting:
    def test_different_tasks_get_different_episodes(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival", "seed": 1})
        state1 = c.get("/state").json()
        c.post("/reset", json={"task_id": "departure", "seed": 2})
        state2 = c.get("/state").json()
        assert state1["task_id"] == "arrival"
        assert state2["task_id"] == "departure"
        assert state1["episode_id"] != state2["episode_id"]

    def test_same_task_and_seed_gives_different_episodes(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "arrival", "seed": 42})
        ep1 = c.get("/state").json()["episode_id"]
        c.post("/reset", json={"task_id": "arrival", "seed": 42})
        ep2 = c.get("/state").json()["episode_id"]
        assert ep1 != ep2
