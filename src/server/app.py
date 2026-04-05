"""OpenEnv server implementing reset/step/state endpoints."""

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.models import Action, Observation, State
from src.openenv_environment import OpenEnvEnvironment

app = FastAPI(title="911 — Dispatch API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_env: OpenEnvEnvironment | None = None


class ResetRequest(BaseModel):
    task_id: str
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict[str, Any]


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request, exc: RuntimeError):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, Any]:
    global _env
    _env = OpenEnvEnvironment(task_id=request.task_id, seed=request.seed)
    obs = await _env.reset()
    return obs.model_dump()


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    if _env is None:
        raise RuntimeError("Environment not initialized. Call /reset first.")
    try:
        action = Action.model_validate(request.action)
    except Exception as e:
        raise RuntimeError(f"Invalid action: {e}")
    obs, reward, done = await _env.step(action)
    return StepResponse(
        observation=obs.model_dump(),
        reward=reward,
        done=done,
    )


@app.get("/state")
async def get_state() -> dict[str, Any]:
    if _env is None:
        raise RuntimeError("Environment not initialized. Call /reset first.")
    state = _env.state()
    return state.model_dump()


@app.get("/dashboard/state")
async def get_dashboard_state() -> dict[str, Any]:
    """Extended state for the HTML live dashboard.

    Keeps the existing /state response stable for typed clients.
    """
    if _env is None:
        raise RuntimeError("Environment not initialized. Call /reset first.")

    state_dict = _env.state().model_dump()
    legal_actions = [a.model_dump() for a in _env.legal_actions()]
    last_obs = _env.last_observation()
    issues = list(last_obs.issues) if last_obs is not None else []

    return {
        **state_dict,
        "legal_actions": legal_actions,
        "issues": issues,
    }


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
