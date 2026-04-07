"""OpenEnv server implementing reset/step/state endpoints."""

from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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


# Removed ResetRequest since /reset now dynamically parses the Request to handle null bodies gracefully.

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


@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    """Root endpoint for Spaces health probes and browser landing."""
    return {
        "status": "healthy",
        "service": "citywide-dispatch-supervisor",
        "health": "/health",
        "tasks": "/tasks",
        "dashboard_state": "/dashboard/state",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    # OpenEnv runtime validation expects status=healthy
    return {"status": "healthy"}


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    """OpenEnv metadata endpoint used by runtime validators."""

    return {
        "name": "citywide-dispatch-supervisor",
        "description": (
            "City-wide 911 emergency dispatch supervisor RL environment. "
            "An LLM agent learns to manage simultaneous incidents by dispatching "
            "police, fire, and EMS units across a city grid under realistic constraints."
        ),
        "version": "0.1.0",
        "mode": "simulation",
    }


@app.get("/schema")
async def schema() -> dict[str, Any]:
    """Return JSON schemas for Action/Observation/State."""

    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": State.model_json_schema(),
    }


@app.post("/mcp")
async def mcp(request: Request) -> dict:
    """Full MCP JSON-RPC endpoint supporting reset/step/state/legal_actions methods."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)

    method = body.get("method", "")
    req_id = body.get("id", 1)

    if method == "reset":
        params = body.get("params", {})
        global _env
        _env = OpenEnvEnvironment(
            task_id=params.get("task_id", "single_incident"),
            seed=params.get("seed"),
        )
        obs = await _env.reset()
        return {"jsonrpc": "2.0", "id": req_id, "result": obs.model_dump()}

    elif method == "step":
        if _env is None:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "Environment not initialized. Call reset first."}},
                status_code=400,
            )
        action_data = body.get("params", {}).get("action", {})
        try:
            action = Action.model_validate(action_data)
        except Exception as e:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Invalid action: {e}"}},
                status_code=400,
            )
        obs, reward, done = await _env.step(action)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"observation": obs.model_dump(), "reward": reward, "done": done},
        }

    elif method == "state":
        if _env is None:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": "Environment not initialized."}},
                status_code=400,
            )
        return {"jsonrpc": "2.0", "id": req_id, "result": _env.state().model_dump()}

    elif method == "legal_actions":
        if _env is None:
            return {"jsonrpc": "2.0", "id": req_id, "result": []}
        actions = _env.legal_actions()
        return {"jsonrpc": "2.0", "id": req_id, "result": [a.model_dump() for a in actions]}

    else:
        # Unknown method — still return 200 with JSON-RPC error (OpenEnv validator just checks reachability)
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


@app.get("/tasks")
async def list_tasks() -> list[dict[str, str]]:
    """List all available tasks."""
    from src.tasks.registry import TaskRegistry

    return [
        {
            "task_id": t.task_id,
            "name": t.name,
            "description": t.description,
            "difficulty": t.difficulty,
        }
        for t in TaskRegistry.list_tasks()
    ]


@app.post("/reset")
async def reset(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        body = {}
    if body is None:
        body = {}
        
    task_id = body.get("task_id", "single_incident")
    seed = body.get("seed", None)

    global _env
    _env = OpenEnvEnvironment(task_id=task_id, seed=seed)
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
        # Return an empty but valid structure before /reset is called
        return {
            "units": {},
            "incidents": {},
            "episode_id": "not-initialized",
            "step_count": 0,
            "task_id": "none",
            "city_time": 0.0,
            "metadata": {},
            "legal_actions": [],
            "issues": [],
            "observation": None,
        }

    state_dict = _env.state().model_dump()
    legal_actions = [a.model_dump() for a in _env.legal_actions()]
    last_obs = _env.last_observation()
    issues = list(last_obs.issues) if last_obs is not None else []
    obs_dict = last_obs.model_dump() if last_obs is not None else None

    return {
        **state_dict,
        "legal_actions": legal_actions,
        "issues": issues,
        "observation": obs_dict,
    }


def main():
    import uvicorn
    import os

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
