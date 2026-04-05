"""REST/API client wrapper for the 911 dispatch environment."""

from __future__ import annotations

from typing import Any

import httpx

from src.models import Action, Observation, State


class APIError(Exception):
    """Raised on non-2xx HTTP responses from the API."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"APIError({status_code}): {detail}")


class DispatchAPI:
    """Async HTTP client for the 911 dispatch environment API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self._client

    async def _close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def reset(self, task_id: str, seed: int | None) -> Observation:
        """Reset the environment and return initial observation."""
        client = self._get_client()
        response = await client.post(
            "/reset",
            json={"task_id": task_id, "seed": seed},
        )
        if response.status_code != 200:
            raise APIError(status_code=response.status_code, detail=response.text)
        data = response.json()
        return Observation.model_validate(data)

    async def step(self, action: Action) -> tuple[Observation, float, bool]:
        """Execute an action and return (observation, reward, done)."""
        client = self._get_client()
        response = await client.post(
            "/step",
            json={"action": action.model_dump()},
        )
        if response.status_code != 200:
            raise APIError(status_code=response.status_code, detail=response.text)
        data = response.json()
        observation = Observation.model_validate(data["observation"])
        reward: float = data["reward"]
        done: bool = data["done"]
        return observation, reward, done

    async def state(self) -> State:
        """Get current environment state."""
        client = self._get_client()
        response = await client.get("/state")
        if response.status_code != 200:
            raise APIError(status_code=response.status_code, detail=response.text)
        data = response.json()
        return State.model_validate(data)

    async def health(self) -> bool:
        """Check if the API server is healthy."""
        client = self._get_client()
        try:
            response = await client.get("/health")
            return response.status_code == 200
        except Exception:
            return False


# Backwards-compatible alias (old ATC name).
ATCAircraftAPI = DispatchAPI
