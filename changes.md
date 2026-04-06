# 911 Dispatch Supervisor — Fix & Polish for OpenEnv Submission

You are working on the repo at the current directory. Apply ALL fixes below in order.
Do not skip any item. After all fixes, run the final validation checklist.

---

## SECTION 1 — CRITICAL BUGS (fix these first)

### 1.1 Fix `openenv.yaml` — Replace entire file content

The file uses hard tab characters which breaks YAML parsing. Replace the entire file with:
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

Verify with: `python -c "import yaml; yaml.safe_load(open('openenv.yaml')); print('YAML OK')"`

---

### 1.2 Fix `src/server/app.py` — Server never starts

Add these two lines at the very bottom of `src/server/app.py`, after the `def main()` block:
```python
if __name__ == "__main__":
    main()
```

Also update the `main()` function to:
```python
def main():
    import uvicorn
    uvicorn.run("src.server.app:app", host="0.0.0.0", port=8000, reload=False)
```

---

### 1.3 Fix `src/server/app.py` — `/reset` rejects empty body

Change `ResetRequest` so `task_id` has a default:
```python
class ResetRequest(BaseModel):
    task_id: str = "single_incident"
    seed: int | None = None
```

---

### 1.4 Fix `Dockerfile` — Use uvicorn directly in CMD

Replace the CMD line in the root `Dockerfile` with:
```dockerfile
CMD ["uv", "run", "uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## SECTION 2 — HIGH PRIORITY BUGS

### 2.1 Fix `validate_local.py` — `check_inference()` never uses random mode

In `validate_local.py`, inside `check_inference()`, add `env["USE_RANDOM"] = "true"` before the `subprocess.run` call:
```python
env["USE_RANDOM"] = "true"
```

Also increase the timeout to 300 seconds if not already set.

---

### 2.2 Fix `pyproject.toml` — Add `asyncio_mode`

In `[tool.pytest.ini_options]`, add:
```toml
asyncio_mode = "auto"
```

---

### 2.3 Fix `inference.py` — Normalize exception error token

In `inference.py`, inside the inner `except Exception as e` block within the step loop, change the error string:
```python
except Exception as e:
    error_msg = "step_error"
    print(
        f"[STEP] step={step_count} action={action_str} "
        f"reward=0.00 done=true error={error_msg}"
    )
    success = False
    break
```

---

### 2.4 Fix `inference.py` — Score computation excludes reset reward

Change score computation to exclude the initial reset observation score:
```python
step_rewards = rewards[1:]
if step_rewards:
    total_score = sum(step_rewards) / len(step_rewards)
else:
    total_score = 0.0
total_score = max(0.0, min(1.0, total_score))

rewards_str = ",".join(f"{r:.2f}" for r in step_rewards) if step_rewards else "0.00"
```

---

### 2.5 Fix `src/server/app.py` — Guard `get_dashboard_state` against None env

The `/dashboard/state` endpoint should return a safe empty structure before `/reset` is called. It already does this in the current code — verify it matches:
```python
@app.get("/dashboard/state")
async def get_dashboard_state() -> dict[str, Any]:
    if _env is None:
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
    # ... rest unchanged
```

---

## SECTION 3 — ENVIRONMENT DESIGN IMPROVEMENTS

### 3.1 Improve `src/tasks/single_incident.py` grader

Replace `SingleIncidentGrader.grade()` with:
```python
def grade(self, state: State, rewards: list[float]) -> float:
    if not rewards:
        return 0.0

    incident = state.incidents.get("INC-001")
    if incident is None:
        return 0.0

    score = 0.0

    if incident.status.value == "RESOLVED":
        score += 0.50

    medic_dispatched = any(
        u.unit_type.value == "MEDIC"
        and (
            u.assigned_incident_id == "INC-001"
            or u.status.value in {"ON_SCENE", "DISPATCHED"}
        )
        for u in state.units.values()
    )
    if medic_dispatched:
        score += 0.30

    if incident.status.value == "RESOLVED" and state.step_count <= 10:
        score += 0.20

    return max(0.0, min(1.0, score))
```

---

### 3.2 Improve `src/tasks/multi_incident.py` grader

Replace `MultiIncidentGrader.grade()` with:
```python
def grade(self, state: State, rewards: list[float]) -> float:
    if not rewards:
        return 0.0

    total = len(state.incidents)
    if total == 0:
        return 0.0

    resolved = sum(1 for i in state.incidents.values() if i.status.value == "RESOLVED")
    failed = sum(1 for i in state.incidents.values() if i.status.value == "ESCALATED")
    p1_total = sum(1 for i in state.incidents.values() if i.severity.value == "PRIORITY_1")
    p1_resolved = sum(
        1
        for iid in state.metadata.get("resolved_incidents", [])
        if state.incidents.get(iid)
        and state.incidents[iid].severity.value == "PRIORITY_1"
    )

    resolution_score = resolved / total
    p1_score = (p1_resolved / p1_total) if p1_total > 0 else 1.0
    failure_penalty = failed / total

    score = 0.5 * p1_score + 0.3 * resolution_score - 0.2 * failure_penalty
    return max(0.0, min(1.0, score))
```

---

### 3.3 Improve `src/tasks/mass_casualty.py` grader

Replace `MassCasualtyGrader.grade()` with:
```python
def grade(self, state: State, rewards: list[float]) -> float:
    if not rewards:
        return 0.0

    p1_seen = list(state.metadata.get("p1_seen", []))
    p1_resolved = [
        iid
        for iid in state.metadata.get("resolved_incidents", [])
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

### 3.4 Fix `src/rewards.py` — Triage key format mismatch

In `_compute_triage()`, the metadata lookup uses inconsistent key formats. Ensure it tries both:
```python
required_types = (
    required_map.get(incident.incident_type.value, [])
    or required_map.get(str(incident.incident_type), [])
)
```

---

### 3.5 Fix `src/state_machine.py` — Use Manhattan distance for ETA

In `_apply_dispatch()`, replace Euclidean distance with Manhattan:
```python
dx = abs(unit.location_x - incident.location_x)
dy = abs(unit.location_y - incident.location_y)
manhattan_dist = dx + dy
eta = manhattan_dist / max(speed, 1e-6)
```

---

## SECTION 4 — TEST FIXES

### 4.1 Update `tests/test_inference.py` — Add `step_error` to valid error tokens

Find `valid_errors` in `test_step_line_error_format` and add `"step_error"`:
```python
valid_errors = {"null", "max_steps_exceeded", "illegal_transition", "step_error"}
```

---

### 4.2 Verify `tests/test_openenv_integration.py` has these two tests

Confirm the following tests exist (they appear to be already present based on the file):
```python
def test_reset_with_empty_body_returns_200(self) -> None:
    c = TestClient(server_app.app)
    response = c.post("/reset", json={})
    assert response.status_code == 200
    data = response.json()
    assert data["result"] == "dispatch center online"

def test_tasks_endpoint_returns_four_tasks(self) -> None:
    c = TestClient(server_app.app)
    response = c.get("/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) == 4
    task_ids = {t["task_id"] for t in tasks}
    assert task_ids == {"single_incident", "multi_incident", "mass_casualty", "shift_surge"}
```

If missing, add them to the `TestTasksEndpoint` and `TestResetEndpoint` classes.

---

## SECTION 5 — FINAL VALIDATION CHECKLIST

Run these commands in order and confirm each passes:
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
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" -d '{}'
docker stop test-dispatch && docker rm test-dispatch

# 8. Benchmark scores all in [0.0, 1.0]
uv run python -c "
from src.benchmark import run_all
scores = run_all()
for task_id, score in scores.items():
    assert 0.0 <= score <= 1.0, f'{task_id}: score {score} out of range'
    print(f'{task_id}: {score:.3f}')
print('All scores in [0.0, 1.0] — PASS')
"
```

All 8 checks must pass before the submission is ready.