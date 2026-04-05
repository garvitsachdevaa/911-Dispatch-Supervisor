# pyright: reportMissingImports=false

"""Tests for ATCAircraftAPI client wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import Action, ClearanceType, LifecyclePhase, Observation, State
from src.api import APIError, ATCAircraftAPI


class TestAPIError:
    """Tests for APIError exception."""

    def test_api_error_has_status_code_and_detail(self) -> None:
        error = APIError(status_code=404, detail="Not found")
        assert error.status_code == 404
        assert error.detail == "Not found"

    def test_api_error_is_raised_on_non_2xx(self) -> None:
        error = APIError(status_code=500, detail="Server error")
        assert isinstance(error, Exception)
        assert error.status_code == 500
        assert error.detail == "Server error"


class TestATCAircraftAPIInit:
    """Tests for ATCAircraftAPI initialization."""

    def test_default_base_url(self) -> None:
        api = ATCAircraftAPI()
        assert api.base_url == "http://localhost:8000"

    def test_custom_base_url(self) -> None:
        api = ATCAircraftAPI(base_url="http://custom:9000")
        assert api.base_url == "http://custom:9000"

    def test_uses_httpx_async_client(self) -> None:
        with patch("src.api.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            api = ATCAircraftAPI()
            api._get_client()
            mock_client_class.assert_called_once()


class TestATCAircraftAPIReset:
    """Tests for ATCAircraftAPI.reset()."""

    @pytest.mark.asyncio
    async def test_reset_returns_observation(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Environment ready",
            "score": 0.0,
            "phraseology_ok": False,
            "issues": [],
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        observation = await api.reset(task_id="arrival", seed=42)

        assert isinstance(observation, Observation)
        assert observation.result == "Environment ready"
        assert observation.score == 0.0
        assert observation.phraseology_ok is False
        assert observation.issues == []

    @pytest.mark.asyncio
    async def test_reset_sends_correct_payload(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Ready",
            "score": 0.0,
            "phraseology_ok": False,
            "issues": [],
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        await api.reset(task_id="departure", seed=123)

        api._client.post.assert_called_once()
        call_args = api._client.post.call_args
        assert call_args[0][0] == "/reset"
        assert call_args[1]["json"] == {"task_id": "departure", "seed": 123}

    @pytest.mark.asyncio
    async def test_reset_with_none_seed(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Ready",
            "score": 0.0,
            "phraseology_ok": False,
            "issues": [],
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        await api.reset(task_id="arrival", seed=None)

        call_kwargs = api._client.post.call_args.kwargs
        assert call_kwargs["json"] == {"task_id": "arrival", "seed": None}

    @pytest.mark.asyncio
    async def test_reset_raises_api_error_on_500(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        with pytest.raises(APIError) as ctx:
            await api.reset(task_id="arrival", seed=42)

        assert ctx.value.status_code == 500
        assert "Internal server error" in ctx.value.detail

    @pytest.mark.asyncio
    async def test_reset_raises_api_error_on_404(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        with pytest.raises(APIError) as ctx:
            await api.reset(task_id="unknown", seed=42)

        assert ctx.value.status_code == 404

    @pytest.mark.asyncio
    async def test_reset_validates_response_shape(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "score": 0.0,
            "phraseology_ok": False,
            "issues": [],
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        with pytest.raises(Exception):
            await api.reset(task_id="arrival", seed=42)


class TestATCAircraftAPIStep:
    """Tests for ATCAircraftAPI.step()."""

    @pytest.mark.asyncio
    async def test_step_returns_tuple(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observation": {
                "result": "Cleared",
                "score": 0.5,
                "phraseology_ok": True,
                "issues": [],
            },
            "reward": 0.5,
            "done": False,
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=[],
        )

        result = await api.step(action)

        assert isinstance(result, tuple)
        assert len(result) == 3
        obs, reward, done = result
        assert isinstance(obs, Observation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    @pytest.mark.asyncio
    async def test_step_sends_action_payload(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "observation": {
                "result": "Cleared",
                "score": 0.0,
                "phraseology_ok": False,
                "issues": [],
            },
            "reward": 0.0,
            "done": False,
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        action = Action(
            clearance_type=ClearanceType.HOLD_SHORT,
            target_callsign="EZY456",
            hold_short=True,
        )

        await api.step(action)

        api._client.post.assert_called_once()
        call_args = api._client.post.call_args
        assert call_args[0][0] == "/step"
        assert call_args[1]["json"]["action"]["clearance_type"] == "hold_short"
        assert call_args[1]["json"]["action"]["target_callsign"] == "EZY456"
        assert call_args[1]["json"]["action"]["hold_short"] is True

    @pytest.mark.asyncio
    async def test_step_raises_api_error_on_400(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Invalid action: clearance_type is required"

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        action = Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign="BAW999",
        )

        with pytest.raises(APIError) as ctx:
            await api.step(action)

        assert ctx.value.status_code == 400

    @pytest.mark.asyncio
    async def test_step_raises_api_error_on_500(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Environment error"

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
        )

        with pytest.raises(APIError) as ctx:
            await api.step(action)

        assert ctx.value.status_code == 500


class TestATCAircraftAPIState:
    """Tests for ATCAircraftAPI.state()."""

    @pytest.mark.asyncio
    async def test_state_returns_state_instance(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "phase": "approach",
            "aircraft": {
                "BAW123": {
                    "callsign": "BAW123",
                    "x_ft": 0.0,
                    "y_ft": 0.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 3000.0,
                    "speed_kt": 250.0,
                    "phase": "approach",
                    "assigned_runway": "27L",
                    "assigned_gate": None,
                    "wake_category": "M",
                }
            },
            "episode_id": "ep-001",
            "step_count": 5,
            "task_id": "arrival",
            "metadata": {},
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        state = await api.state()

        assert isinstance(state, State)
        assert state.phase == LifecyclePhase.APPROACH
        assert state.episode_id == "ep-001"
        assert state.step_count == 5
        assert "BAW123" in state.aircraft

    @pytest.mark.asyncio
    async def test_state_sends_correct_url(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "phase": "approach",
            "aircraft": {},
            "episode_id": "ep-001",
            "step_count": 0,
            "task_id": "arrival",
            "metadata": {},
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        await api.state()

        api._client.get.assert_called_once()
        call_args = api._client.get.call_args
        assert call_args[0][0] == "/state"

    @pytest.mark.asyncio
    async def test_state_raises_api_error_on_500(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        with pytest.raises(APIError) as ctx:
            await api.state()

        assert ctx.value.status_code == 500

    @pytest.mark.asyncio
    async def test_state_raises_api_error_on_404(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        with pytest.raises(APIError) as ctx:
            await api.state()

        assert ctx.value.status_code == 404


class TestATCAircraftAPIHealth:
    """Tests for ATCAircraftAPI.health()."""

    @pytest.mark.asyncio
    async def test_health_returns_bool(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        result = await api.health()

        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_health_returns_false_on_500(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 500

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        result = await api.health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_returns_false_on_timeout(self) -> None:
        import asyncio

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(side_effect=asyncio.TimeoutError)
        api._client.aclose = AsyncMock()

        result = await api.health()

        assert result is False

    @pytest.mark.asyncio
    async def test_health_sends_correct_url(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.get = AsyncMock(return_value=mock_response)
        api._client.aclose = AsyncMock()

        await api.health()

        api._client.get.assert_called_once()
        call_args = api._client.get.call_args
        assert call_args[0][0] == "/health"


class TestEndToEndFlow:
    """End-to-end flow tests for ATCAircraftAPI."""

    @pytest.mark.asyncio
    async def test_full_reset_step_state_flow(self) -> None:
        reset_response = MagicMock()
        reset_response.status_code = 200
        reset_response.json.return_value = {
            "result": "Environment ready",
            "score": 0.0,
            "phraseology_ok": False,
            "issues": [],
        }

        step_response = MagicMock()
        step_response.status_code = 200
        step_response.json.return_value = {
            "observation": {
                "result": "Cleared to taxi",
                "score": 0.1,
                "phraseology_ok": True,
                "issues": [],
            },
            "reward": 0.1,
            "done": False,
        }

        state_response = MagicMock()
        state_response.status_code = 200
        state_response.json.return_value = {
            "phase": "taxi_in",
            "aircraft": {
                "BAW123": {
                    "callsign": "BAW123",
                    "x_ft": 100.0,
                    "y_ft": 200.0,
                    "heading_deg": 90.0,
                    "altitude_ft": 0.0,
                    "speed_kt": 20.0,
                    "phase": "taxi_in",
                    "assigned_runway": None,
                    "assigned_gate": "A1",
                    "wake_category": "M",
                }
            },
            "episode_id": "ep-001",
            "step_count": 1,
            "task_id": "arrival",
            "metadata": {},
        }

        api = ATCAircraftAPI()
        api._client = AsyncMock()
        api._client.post = AsyncMock(side_effect=[reset_response, step_response])
        api._client.get = AsyncMock(return_value=state_response)
        api._client.aclose = AsyncMock()

        obs = await api.reset(task_id="arrival", seed=42)
        assert isinstance(obs, Observation)
        assert obs.result == "Environment ready"

        action = Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign="BAW123",
            route=["alpha", "bravo"],
        )
        obs2, reward, done = await api.step(action)
        assert isinstance(obs2, Observation)
        assert reward == 0.1
        assert done is False

        state = await api.state()
        assert isinstance(state, State)
        assert state.phase == LifecyclePhase.TAXI_IN
        assert state.step_count == 1
