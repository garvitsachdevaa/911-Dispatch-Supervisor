"""Tests for DispatchAPI client wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api import APIError, ATCAircraftAPI, DispatchAPI
from src.models import Action, DispatchAction, Observation, State


class TestAPIError:
    def test_fields(self) -> None:
        err = APIError(status_code=404, detail="Not found")
        assert err.status_code == 404
        assert "Not found" in err.detail


class TestDispatchAPIInit:
    def test_default_base_url(self) -> None:
        api = DispatchAPI()
        assert api.base_url == "http://localhost:8000"

    def test_alias_exists(self) -> None:
        api = ATCAircraftAPI()
        assert isinstance(api, DispatchAPI)

    def test_uses_httpx_async_client(self) -> None:
        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            api = DispatchAPI()
            api._get_client()
            mock_client_class.assert_called_once()


class TestDispatchAPIReset:
    def test_reset_returns_observation(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "dispatch center online",
            "score": 0.0,
            "protocol_ok": True,
            "issues": [],
        }

        api = DispatchAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)

        obs = asyncio.run(api.reset(task_id="single_incident", seed=42))
        assert isinstance(obs, Observation)
        assert obs.protocol_ok is True

    def test_reset_raises_on_non_200(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "boom"

        api = DispatchAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)

        with pytest.raises(APIError):
            asyncio.run(api.reset(task_id="single_incident", seed=1))


class TestDispatchAPIStep:
    def test_step_sends_action_payload(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observation": {
                "result": "ok",
                "score": 0.8,
                "protocol_ok": True,
                "issues": [],
            },
            "reward": 0.8,
            "done": False,
        }

        api = DispatchAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)

        action = Action(
            action_type=DispatchAction.DISPATCH,
            unit_id="MED-1",
            incident_id="INC-001",
        )
        obs, reward, done = asyncio.run(api.step(action))

        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

        call_kwargs = api._client.post.call_args.kwargs
        assert call_kwargs["json"]["action"]["action_type"] == "DISPATCH"
        assert call_kwargs["json"]["action"]["unit_id"] == "MED-1"


class TestDispatchAPIState:
    def test_state_returns_state(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "units": {},
            "incidents": {},
            "episode_id": "ep",
            "step_count": 0,
            "task_id": "single_incident",
            "city_time": 0.0,
            "metadata": {},
        }

        api = DispatchAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)

        state = asyncio.run(api.state())
        assert isinstance(state, State)
        assert state.task_id == "single_incident"
