"""Tests for OpenEnv server/client integration (dispatch domain)."""

from __future__ import annotations

import asyncio
import pytest
from fastapi.testclient import TestClient

import src.server.app as server_app
from src.models import Action, DispatchAction
from src.openenv_environment import OpenEnvEnvironment


@pytest.fixture(autouse=True)
def reset_env() -> None:
    server_app._env = None
    yield
    server_app._env = None


class TestOpenEnvEnvironment:
    def test_reset_and_state(self) -> None:
        env = OpenEnvEnvironment(task_id="single_incident", seed=42)
        obs = asyncio.run(env.reset())
        assert obs.result == "dispatch center online"
        assert obs.protocol_ok is True

        st = env.state()
        assert st.task_id == "single_incident"
        assert st.step_count == 0

    def test_step_returns_tuple(self) -> None:
        env = OpenEnvEnvironment(task_id="single_incident", seed=42)
        asyncio.run(env.reset())
        action = Action(
            action_type=DispatchAction.DISPATCH,
            unit_id="MED-1",
            incident_id="INC-001",
        )
        obs, reward, done = asyncio.run(env.step(action))
        assert isinstance(obs.result, str)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        env.close()


class TestResetEndpoint:
    def test_reset_returns_observation(self) -> None:
        c = TestClient(server_app.app)
        response = c.post("/reset", json={"task_id": "single_incident", "seed": 42})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "dispatch center online"
        assert data["protocol_ok"] is True

    def test_reset_with_empty_body_returns_200(self) -> None:
        """Verify prevalidation.sh compatible: POST /reset with {} returns 200."""
        c = TestClient(server_app.app)
        response = c.post("/reset", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "dispatch center online"


class TestStepEndpoint:
    def test_step_requires_reset_first(self) -> None:
        c = TestClient(server_app.app)
        response = c.post(
            "/step",
            json={
                "action": {
                    "action_type": "DISPATCH",
                    "unit_id": "MED-1",
                    "incident_id": "INC-001",
                }
            },
        )
        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()

    def test_step_invalid_action_rejected(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "single_incident", "seed": 42})
        response = c.post("/step", json={"action": {"invalid": "field"}})
        assert response.status_code == 500
        assert "invalid action" in response.json()["detail"].lower()

    def test_step_ok(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "single_incident", "seed": 42})
        response = c.post(
            "/step",
            json={
                "action": {
                    "action_type": "DISPATCH",
                    "unit_id": "MED-1",
                    "incident_id": "INC-001",
                }
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"observation", "reward", "done"}


class TestStateEndpoint:
    def test_state_requires_reset_first(self) -> None:
        c = TestClient(server_app.app)
        response = c.get("/state")
        assert response.status_code == 500

    def test_state_returns_current_state(self) -> None:
        c = TestClient(server_app.app)
        c.post("/reset", json={"task_id": "single_incident", "seed": 42})
        response = c.get("/state")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "single_incident"
        assert data["step_count"] == 0


class TestHealthEndpoint:
    def test_health_ok(self) -> None:
        c = TestClient(server_app.app)
        response = c.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestTasksEndpoint:
    def test_tasks_endpoint_returns_four_tasks(self) -> None:
        c = TestClient(server_app.app)
        response = c.get("/tasks")
        assert response.status_code == 200
        tasks = response.json()
        assert len(tasks) == 4
        task_ids = {t["task_id"] for t in tasks}
        assert task_ids == {
            "single_incident",
            "multi_incident",
            "mass_casualty",
            "shift_surge",
        }


class TestDashboardEndpoint:
    def test_dashboard_state_before_reset_returns_valid_shape(self) -> None:
        c = TestClient(server_app.app)
        response = c.get("/dashboard/state")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == "none"
        assert data["step_count"] == 0
        assert isinstance(data["units"], dict)
        assert isinstance(data["incidents"], dict)
        assert isinstance(data["legal_actions"], list)
        assert isinstance(data["issues"], list)
        assert data["observation"] is None

    def test_dashboard_state_after_reset_exposes_legal_actions(self) -> None:
        c = TestClient(server_app.app)
        reset_response = c.post("/reset", json={"task_id": "single_incident", "seed": 42})
        assert reset_response.status_code == 200

        response = c.get("/dashboard/state")
        assert response.status_code == 200

        data = response.json()
        assert data["task_id"] == "single_incident"
        assert isinstance(data["legal_actions"], list)
        assert data["observation"] is not None
