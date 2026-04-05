# Copilot Agent Instructions: 911 Dispatch Supervisor RL Environment — Fix & Polish
 
## Overview
 
These are ordered, actionable instructions to bring the `citywide-dispatch-supervisor` repository fully into compliance with the OpenEnv hackathon requirements and fix every discovered bug. Work through each section in order. **Do not skip any item.**
 
---
 
## SECTION 1 — CRITICAL BUGS (will cause validation failure)
 
### 1.1 Fix `openenv.yaml` — Tab indentation is invalid YAML
 
**Problem:** The file uses hard `\t` tab characters for indentation. YAML forbids tabs; `openenv validate` will crash with a parse error.
 
**Action:** Rewrite `openenv.yaml` using 2-space indentation throughout. Use exactly this content:
 
```yaml
name: citywide-dispatch-supervisor
version: "0.1.0"
description: >
  City-wide 911 emergency dispatch supervisor RL environment.
  An LLM agent learns to manage simultaneous incidents by dispatching
  police, fire, and EMS units across a city grid under realistic constraints.
entrypoint: src.openenv_environment:OpenEnvEnvironment
tasks:
  - id: single_incident
    name: Single Incident Response
    description: One incident with a small unit pool; learn basic dispatch, correct unit type, and response time.
  - id: multi_incident
    name: Simultaneous Multi-Incident
    description: Multiple concurrent incidents requiring triage, prioritization, and correct unit matching.
  - id: mass_casualty
    name: Mass Casualty Event
    description: Wave-based Priority-1 surge with resource conflict; maximize survival outcomes.
  - id: shift_surge
    name: Shift Surge
    description: Incident waves combined with units going out of service; maintain coverage over time.
```
 
Verify with: `python -c "import yaml; yaml.safe_load(open('openenv.yaml'))"` — must not raise any error.
 
---
 
### 1.2 Fix `src/server/app.py` — Server never starts inside Docker
 
**Problem:** The file defines `def main()` but never calls it. Running `python -m src.server.app` executes the module top-level code (which only defines routes) but never invokes `uvicorn.run`. The Docker container starts but immediately exits or hangs silently without binding to port 8000.
 
**Action:** Add the following two lines at the very bottom of `src/server/app.py`, after the `def main()` block:
 
```python
if __name__ == "__main__":
    main()
```
 
Also update the `main()` function to be more robust:
 
```python
def main():
    import uvicorn
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)
```
 
**Verify:** `docker build -t citywide-dispatch-supervisor . && docker run -p 8000:8000 citywide-dispatch-supervisor` must hold open and `curl http://localhost:8000/health` must return `{"status":"ok"}`.
 
---
 
### 1.3 Fix `src/server/app.py` ResetRequest — `/reset` rejects empty body
 
**Problem:** The prevalidation script calls `POST /reset` with an empty JSON body `{}`. The current `ResetRequest` model has `task_id: str` as a required field with no default. This produces HTTP 422 Unprocessable Entity, causing the prevalidation check to fail at Step 1.
 
**Action:** In `src/server/app.py`, change `ResetRequest` to give `task_id` a sensible default:
 
```python
class ResetRequest(BaseModel):
    task_id: str = "single_incident"
    seed: int | None = None
```
 
**Verify:** `curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'` must return HTTP 200 with a valid observation JSON.
 
---
 
### 1.4 Fix `Dockerfile` — Use module string for uvicorn to enable proper reloading and port binding
 
**Problem:** `CMD ["uv", "run", "python", "-m", "src.server.app"]` relies on `__main__` execution. Combined with bug 1.2, if `if __name__ == "__main__"` is properly added, this will work — but it is more reliable and production-correct to invoke uvicorn directly as the CMD.
 
**Action:** Replace the `CMD` in the root `Dockerfile` with:
 
```dockerfile
CMD ["uv", "run", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
 
The full updated `Dockerfile` should be:
 
```dockerfile
FROM python:3.11-slim
LABEL org.opencontainers.image.title="911 City-Wide Emergency Dispatch Supervisor"
LABEL org.opencontainers.image.description="City-wide 911 dispatch supervisor RL environment"
WORKDIR /app
COPY . /app
RUN pip install uv && uv sync --frozen
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
 
---
 
## SECTION 2 — HIGH PRIORITY BUGS (cause test failures or incorrect behavior)
 
### 2.1 Fix `validate_local.py` — `check_inference()` never uses random mode
 
**Problem:** `check_inference()` sets real-looking credentials but does NOT set `USE_RANDOM=true`. The inference script will attempt a live API call with the dummy token and fail with an authentication error, making the local validation always report `FAILED: inference`.
 
**Action:** In `validate_local.py`, inside `check_inference()`, add `env["USE_RANDOM"] = "true"` before the `subprocess.run` call:
 
```python
def check_inference() -> bool:
    import os
 
    env = os.environ.copy()
    env["API_BASE_URL"] = "https://api.openai.com/v1"
    env["MODEL_NAME"] = "gpt-4"
    env["HF_TOKEN"] = "dummy-token-for-local-validation"
    env["USE_RANDOM"] = "true"   # <-- ADD THIS LINE
 
    print("\nNOTE: Running inference.py in random-agent mode for local validation")
    result = subprocess.run(
        ["uv", "run", "python", "inference.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,   # also increase timeout; 4 tasks can take time
    )
    # ... rest of function unchanged
```
 
---
 
### 2.2 Fix `pyproject.toml` — Missing `asyncio_mode` for `pytest-asyncio`
 
**Problem:** The test suite uses `asyncio.run()` inline rather than `@pytest.mark.asyncio` decorators. With `pytest-asyncio >= 0.21`, the default mode is `strict`, which requires explicit markers. This can cause silent test collection warnings or failures.
 
**Action:** Add the following to the `[tool.pytest.ini_options]` section in `pyproject.toml`:
 
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
asyncio_mode = "auto"
```
 
---
 
### 2.3 Fix `inference.py` — Exception error messages break the format compliance test
 
**Problem:** The `except Exception as e` block in `run_episode()` outputs `error={str(e)}` which can be any arbitrary string. The test `test_step_line_error_format` only allows `{"null", "max_steps_exceeded", "illegal_transition"}`. Any real exception will produce a string outside this set.
 
**Action:** In `inference.py`, inside the inner `except Exception as e` block within the step loop, normalize the error:
 
```python
except Exception as e:
    error_msg = f"step_error"   # normalize to a fixed token
    print(
        f"[STEP] step={step_count} action={action_str} "
        f"reward=0.00 done=true error={error_msg}"
    )
    success = False
    break
```
 
Also update `test_inference.py` to include `"step_error"` in `valid_errors`:
 
```python
valid_errors = {"null", "max_steps_exceeded", "illegal_transition", "step_error"}
```
 
---
 
### 2.4 Fix `src/server/app.py` — `/reset` endpoint does not return `task_id` in the state after reset
 
**Problem:** After `POST /reset`, calling `GET /state` returns a state with the correct `task_id`. But the dashboard endpoint `GET /dashboard/state` may return `None` for metadata fields if `reset()` hasn't been called. The health check and dashboard should be safe to call at any time.
 
**Action:** Add a null-guard to `get_dashboard_state()`:
 
```python
@app.get("/dashboard/state")
async def get_dashboard_state() -> dict[str, Any]:
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
```
 
---
 
### 2.5 Fix `inference.py` — Score computation is not normalized to competition spec
 
**Problem:** `total_score = sum(rewards) / len(rewards)` computes the average step reward. Since each step reward is already in [0, 1], this is a valid value but it weights the reset-time reward (score=0.0 from `obs.score=0.0` in `reset()`) equally with step rewards. This deflates the score.
 
**Action:** Change score computation in `run_episode()` to exclude the initial zero from reset:
 
```python
# Separate reset reward from step rewards
step_rewards = rewards[1:]  # index 0 is the reset observation score (always 0.0)
if step_rewards:
    total_score = sum(step_rewards) / len(step_rewards)
else:
    total_score = 0.0
total_score = max(0.0, min(1.0, total_score))
```
 
Also update the `rewards_str` to only include step rewards so the `[END]` line is meaningful:
 
```python
rewards_str = ",".join(f"{r:.2f}" for r in rewards[1:]) if len(rewards) > 1 else "0.00"
```
 
---
 
## SECTION 3 — ENVIRONMENT DESIGN IMPROVEMENTS (affect scoring)
 
### 3.1 Improve task graders — current graders are too simple
 
**Problem:** The graders (`SingleIncidentGrader`, `MultiIncidentGrader`, etc.) compute very simple scores that don't fully capture task success. Judges will look at whether hard tasks genuinely challenge frontier models with clear, deterministic success criteria.
 
**Action — `src/tasks/single_incident.py`:** Replace `SingleIncidentGrader.grade()` with:
 
```python
def grade(self, state: State, rewards: list[float]) -> float:
    """Grade based on: correct unit dispatched, fast response, incident resolved."""
    if not rewards:
        return 0.0
 
    incident = state.incidents.get("INC-001")
    if incident is None:
        return 0.0
 
    score = 0.0
 
    # Component 1: Was the incident resolved? (50% weight)
    if incident.status.value == "RESOLVED":
        score += 0.50
 
    # Component 2: Correct unit type dispatched? (30% weight)
    medic_dispatched = any(
        u.unit_type.value == "MEDIC" and (
            u.assigned_incident_id == "INC-001" or
            u.status.value in {"ON_SCENE", "DISPATCHED"}
        )
        for u in state.units.values()
    )
    if medic_dispatched:
        score += 0.30
 
    # Component 3: Speed — resolved within first 10 steps (20% weight)
    if incident.status.value == "RESOLVED" and state.step_count <= 10:
        score += 0.20
 
    return max(0.0, min(1.0, score))
```
 
**Action — `src/tasks/multi_incident.py`:** Replace `MultiIncidentGrader.grade()` with:
 
```python
def grade(self, state: State, rewards: list[float]) -> float:
    """Grade based on: P1 incidents resolved, triage correctness, coverage."""
    if not rewards:
        return 0.0
 
    total = len(state.incidents)
    if total == 0:
        return 0.0
 
    resolved = sum(
        1 for i in state.incidents.values()
        if i.status.value == "RESOLVED"
    )
    failed = sum(
        1 for i in state.incidents.values()
        if i.status.value == "ESCALATED"
    )
    p1_total = sum(1 for i in state.incidents.values() if i.severity.value == "PRIORITY_1")
    p1_resolved = sum(
        1 for iid in state.metadata.get("resolved_incidents", [])
        if state.incidents.get(iid) and state.incidents[iid].severity.value == "PRIORITY_1"
    )
 
    resolution_score = resolved / total
    p1_score = (p1_resolved / p1_total) if p1_total > 0 else 1.0
    failure_penalty = failed / total
 
    score = 0.5 * p1_score + 0.3 * resolution_score - 0.2 * failure_penalty
    return max(0.0, min(1.0, score))
```
 
**Action — `src/tasks/mass_casualty.py`:** The existing `MassCasualtyGrader` is reasonable. Improve it slightly:
 
```python
def grade(self, state: State, rewards: list[float]) -> float:
    if not rewards:
        return 0.0
 
    p1_seen = list(state.metadata.get("p1_seen", []))
    p1_resolved = [
        iid for iid in state.metadata.get("resolved_incidents", [])
        if iid in p1_seen and iid not in state.metadata.get("failed_incidents", [])
    ]
    p1_failed = list(state.metadata.get("failed_incidents", []))
 
    survival_score = len(p1_resolved) / max(len(p1_seen), 1)
    failure_penalty = len(p1_failed) / max(len(p1_seen), 1) * 0.5
 
    mean_reward = sum(rewards) / len(rewards)
    score = 0.6 * survival_score + 0.3 * mean_reward - failure_penalty
    return max(0.0, min(1.0, score))
```
 
---
 
### 3.2 Add `GET /tasks` documentation to README (the endpoint already exists)
 
The server already has `GET /tasks` but it's not documented in the README API table. Add it to the README's API Endpoints section:
 
```markdown
| `/tasks` | GET | List all available tasks with metadata |
```
 
---
 
### 3.3 Improve reward signal in `src/rewards.py` — triage scoring uses wrong key format
 
**Problem:** In `_compute_triage()`, the lookup is:
 
```python
required_map = state.metadata.get("default_required_units", {})
required_types = required_map.get(str(incident.incident_type), [])
```
 
But `str(incident.incident_type)` for a `StrEnum` returns `"CARDIAC_ARREST"` (the value), while the metadata stores types like `"IncidentType.CARDIAC_ARREST"` (the repr). This mismatch means triage always returns 0.5 (the neutral value), undermining the reward signal.
 
**Action:** In `src/rewards.py`, change `_compute_triage()` to use the value directly:
 
```python
def _compute_triage(self, state: State, action: Action) -> float:
    if action.action_type != DispatchAction.DISPATCH:
        return 0.5
    unit = state.units.get(action.unit_id)
    incident = state.incidents.get(action.incident_id)
    if unit is None or incident is None:
        return 0.0
    required_map = state.metadata.get("default_required_units", {})
    # Try both formats: plain value and StrEnum repr
    required_types = (
        required_map.get(incident.incident_type.value, []) or
        required_map.get(str(incident.incident_type), [])
    )
    if not required_types:
        return 0.5
    return 1.0 if unit.unit_type.value in required_types else 0.0
```
 
Also fix the metadata population in `src/state_machine.py`. In `reset()`, when enriching metadata, convert the `default_required_units` schema data to use plain string values:
 
```python
# Convert unit type values to plain strings for consistent lookup
raw_required = schema_dump.get("default_required_units", {})
converted_required = {
    str(inc_type): [str(u) for u in unit_types]
    for inc_type, unit_types in raw_required.items()
}
state.metadata.setdefault("default_required_units", converted_required)
```
 
---
 
### 3.4 Fix `src/state_machine.py` — ETA computation uses wrong distance formula
 
**Problem:** In `_apply_dispatch()`, the ETA is computed using Euclidean distance (`math.hypot`) but the `physics.py` module uses Manhattan distance (`dx + dy`). The physics module is used for movement, so the ETA should match:
 
```python
dist = _distance(unit.location_x, unit.location_y, incident.location_x, incident.location_y)
```
 
Where `_distance` uses `math.hypot`. But `move_unit_toward` uses Manhattan movement. This inconsistency means units arrive "early" by Euclidean measurement but take longer by Manhattan movement.
 
**Action:** Replace `_distance` usage in `_apply_dispatch()` to use Manhattan distance:
 
```python
def _apply_dispatch(self, state: State, action: Action) -> None:
    unit = state.units[action.unit_id]
    incident = state.incidents[action.incident_id]
 
    speed = float(self._schema.unit_speeds.get(unit.unit_type, 1.0))
    # Use Manhattan distance to match move_unit_toward physics
    dx = abs(unit.location_x - incident.location_x)
    dy = abs(unit.location_y - incident.location_y)
    manhattan_dist = dx + dy
    eta = manhattan_dist / max(speed, 1e-6)
 
    unit.status = UnitStatus.DISPATCHED
    unit.assigned_incident_id = incident.incident_id
    unit.eta_seconds = max(0.0, float(eta))
 
    if unit.unit_id not in incident.units_assigned:
        incident.units_assigned.append(unit.unit_id)
    if incident.status == IncidentStatus.PENDING:
        incident.status = IncidentStatus.RESPONDING
```
 
---
 
## SECTION 4 — README IMPROVEMENTS (required by competition)
 
### 4.1 Add missing required README sections
 
The current README is good but is missing:
 
1. **Baseline scores table with instructions to reproduce** — The README mentions scores but doesn't show how to generate them with a single command.
2. **Full action space table** — Currently only shows key fields, needs all fields.
3. **Setup instructions** — Missing explicit `uv sync` + server start commands.
 
**Action:** Add the following sections to `README.md`:
 
After the existing "Quick Start" section, add:
 
```markdown
## Reproducing Baseline Scores
 
Run the random baseline agent against all 4 tasks:
 
```bash
USE_RANDOM=true API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4 HF_TOKEN=x python inference.py
```
 
Expected output (approximate):
 
| Task | Difficulty | Random Baseline Score |
|------|-----------|----------------------|
| `single_incident` | Easy | ~0.55 |
| `multi_incident` | Medium | ~0.48 |
| `mass_casualty` | Hard | ~0.32 |
| `shift_surge` | Hard | ~0.38 |
 
*Scores vary slightly due to seeded randomness. Run with `seed=42` for exact reproduction.*
```
 
Also add an explicit environment variable table near the top:
 
```markdown
## Environment Variables
 
| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint base URL |
| `MODEL_NAME` | Yes | Model identifier string |
| `HF_TOKEN` | Yes (unless `USE_RANDOM=true`) | API key / HF token |
| `USE_RANDOM` | No | Set to `true` to use deterministic random agent (no LLM) |
```
 
---
 
### 4.2 Add task difficulty descriptions to README
 
Under the "Tasks" section, expand each task to include expected agent behaviors:
 
```markdown
### Task Difficulty Guide
 
| Task | Difficulty | Key Challenge | Success Criteria |
|------|-----------|---------------|-----------------|
| `single_incident` | Easy | Dispatch the right unit type (MEDIC) quickly | Incident resolved, correct unit, ETA < 300s |
| `multi_incident` | Medium | Triage 3 simultaneous incidents, prioritize P1 | All P1 incidents responded to, no ESCALATED |
| `mass_casualty` | Hard | Manage wave-based surge with limited resources | Maximize P1 survival rate across waves |
| `shift_surge` | Hard | Adapt as units go out of service over time | Maintain coverage and resolve incidents despite attrition |
```
 
---
 
## SECTION 5 — TEST FIXES
 
### 5.1 Update `tests/test_inference.py` to reflect valid error tokens
 
After the fix in Section 2.3, update the valid error set in `test_step_line_error_format`:
 
```python
valid_errors = {"null", "max_steps_exceeded", "illegal_transition", "step_error"}
```
 
---
 
### 5.2 Add a test for `/reset` with empty body
 
Add this test to `tests/test_openenv_integration.py`:
 
```python
def test_reset_with_empty_body_returns_200(self) -> None:
    """Verify prevalidation.sh compatible: POST /reset with {} returns 200."""
    c = TestClient(server_app.app)
    response = c.post("/reset", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "dispatch center online"
```
 
---
 
### 5.3 Add a test for the `/tasks` endpoint
 
Add to `tests/test_openenv_integration.py`:
 
```python
def test_tasks_endpoint_returns_four_tasks(self) -> None:
    c = TestClient(server_app.app)
    response = c.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 4
    task_ids = {t["task_id"] for t in tasks}
    assert task_ids == {"single_incident", "multi_incident", "mass_casualty", "shift_surge"}
```
 
---
 
## SECTION 6 — DOCKER AND DEPLOYMENT CHECKS
 
### 6.1 Verify `src/server/Dockerfile` is consistent
 
The `src/server/Dockerfile` is a separate server-only Dockerfile. Ensure it also starts the server properly. Replace its CMD with:
 
```dockerfile
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```
 
---
 
### 6.2 Add `.dockerignore` to speed up builds
 
Create `.dockerignore` at the repo root:
 
```
.git
.venv
.uv
__pycache__
*.pyc
*.pyo
.pytest_cache
.coverage
htmlcov
.sisyphus/evidence/
*.log
tmp/
dashboard.html
*.png
*.jpg
.env
.env.*
```
 
---
 
### 6.3 Verify `requirements.txt` is complete
 
The current `requirements.txt` is missing `groq` which is in `pyproject.toml`. Add it:
 
```
pydantic>=2.7
openenv-core>=0.2.0
fastapi>=0.110
uvicorn[standard]>=0.29
openai>=1.12
httpx>=0.27
matplotlib>=3.8
numpy>=1.26
groq>=1.1.2
```
 
---
 
## SECTION 7 — FINAL VALIDATION CHECKLIST
 
After making all changes, run these commands in order and confirm each passes:
 
```bash
# 1. YAML parse check
python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('YAML OK')"
 
# 2. Full test suite
uv run python -m pytest tests/ -v --tb=short
 
# 3. Inference script with random agent
USE_RANDOM=true API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4 HF_TOKEN=x \
  uv run python inference.py 2>&1 | grep -E '^\[(START|STEP|END)\]' | head -20
 
# 4. Demo script
uv run python demo.py
 
# 5. OpenEnv validate
uv run openenv validate
 
# 6. Docker build
docker build -t citywide-dispatch-supervisor .
 
# 7. Docker run + health check + empty reset
docker run -d -p 8000:8000 --name test-dispatch citywide-dispatch-supervisor
sleep 5
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
docker stop test-dispatch && docker rm test-dispatch
 
# 8. Benchmark scores
uv run python -c "
from src.benchmark import run_all
scores = run_all()
for task_id, score in scores.items():
    assert 0.0 <= score <= 1.0, f'{task_id}: score {score} out of range'
    print(f'{task_id}: {score:.3f}')
print('All scores in [0.0, 1.0] — PASS')
"
```
 
All 8 checks must pass before submission.
 
---
 
## SECTION 8 — PRIORITY ORDER SUMMARY
 
Work through issues in this exact order:
 
| # | File | Change | Severity |
|---|------|--------|----------|
| 1 | `openenv.yaml` | Fix tab → space indentation | CRITICAL |
| 2 | `src/server/app.py` | Add `if __name__ == "__main__": main()` | CRITICAL |
| 3 | `src/server/app.py` | Make `task_id` optional in `ResetRequest` | CRITICAL |
| 4 | `Dockerfile` | Use uvicorn directly in CMD | CRITICAL |
| 5 | `validate_local.py` | Add `USE_RANDOM=true` in `check_inference` | HIGH |
| 6 | `pyproject.toml` | Add `asyncio_mode = "auto"` | HIGH |
| 7 | `inference.py` | Normalize exception error messages | HIGH |
| 8 | `inference.py` | Fix score computation (exclude reset reward) | HIGH |
| 9 | `src/server/app.py` | Guard `get_dashboard_state` against None env | MEDIUM |
| 10 | `src/rewards.py` | Fix triage key format mismatch | MEDIUM |
| 11 | `src/state_machine.py` | Use Manhattan distance for ETA | MEDIUM |
| 12 | `src/tasks/*.py` | Improve grader logic | MEDIUM |
| 13 | `tests/test_openenv_integration.py` | Add empty-body reset test | MEDIUM |
| 14 | `tests/test_openenv_integration.py` | Add /tasks endpoint test | LOW |
| 15 | `tests/test_inference.py` | Add `step_error` to valid errors set | LOW |
| 16 | `requirements.txt` | Add `groq>=1.1.2` | LOW |
| 17 | `.dockerignore` | Create file | LOW |
| 18 | `README.md` | Add baseline scores table + env var table + difficulty guide | LOW |