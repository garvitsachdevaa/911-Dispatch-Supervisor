---
title: 911 Dispatch Supervisor
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - llm-agent
  - emergency-dispatch
---

# 🚨 911 Dispatch Supervisor

> **A city-wide emergency dispatch RL environment** — train and evaluate LLM agents to manage simultaneous incidents by dispatching police, fire, and EMS units across a city grid under realistic resource constraints.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://openenv.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://hub.docker.com)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Overview

The **911 Dispatch Supervisor** models real-world emergency dispatch operations. At every step, an LLM agent plays the role of a city-wide dispatch supervisor, deciding which units to dispatch, reassign, cancel, stage, or escalate — under time pressure, limited resources, and competing priorities.

This is not a toy environment. Emergency dispatch is a high-stakes, multi-objective decision problem that:

- Requires triage (prioritizing life-threatening incidents over property damage)
- Demands coverage awareness (keeping geographic zones protected)
- Rewards correct unit-type matching (sending a MEDIC vs. an ENGINE)
- Punishes delays that cause Priority-1 incidents to escalate

### Why This Domain?

Real-world 911 dispatch centers field thousands of concurrent calls daily. Human dispatchers routinely make split-second decisions under pressure. Modeling this as an RL environment enables:

- **Benchmarking** frontier LLM judgment under operational stress
- **Training** agents on triage and multi-constraint resource allocation
- **Evaluating** decision quality against programmatic, real-world-grounded graders

---

## Environment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   OpenEnv Interface                     │
│         reset() · step(action) · state()                │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              DispatchStateMachine                       │
│  • Validates actions via DispatchProtocolValidator      │
│  • Moves units toward incidents (Manhattan physics)     │
│  • Advances incident status: PENDING → RESPONDING →     │
│    ON_SCENE → RESOLVED (or ESCALATED if timeout)        │
│  • Spawns incident waves at configured step offsets     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  RewardCalculator                       │
│  • response_time (30%) · triage (25%) · survival (25%) │
│  • coverage (12%) · protocol (8%)                       │
│  • Safety gate: P1 failure → score capped at 0.2        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│            Task-Specific Episode Graders                │
│  single_incident · multi_incident · mass_casualty ·     │
│                   shift_surge                           │
└─────────────────────────────────────────────────────────┘
```

---

## Action Space

Actions are structured Pydantic models — no free-text parsing required.

**`src.models.Action`**

| Field | Type | Description |
|---|---|---|
| `action_type` | `DispatchAction` | One of: `DISPATCH`, `CANCEL`, `REASSIGN`, `STAGE`, `MUTUAL_AID`, `UPGRADE`, `DOWNGRADE` |
| `unit_id` | `str` | Unit identifier, e.g. `MED-1`, `ENG-2` |
| `incident_id` | `str` | Incident identifier, e.g. `INC-001` |
| `notes` | `str \| None` | Optional phraseology text for protocol scoring bonus |
| `priority_override` | `IncidentSeverity \| None` | Required for `UPGRADE`/`DOWNGRADE` actions |

**Action Types**

| Action | Description | Protocol Rule |
|---|---|---|
| `DISPATCH` | Send an available unit to an incident | Unit must be `AVAILABLE`; incident must not be `RESOLVED` |
| `CANCEL` | Release a unit from its current assignment | Unit must be assigned to the specified incident |
| `REASSIGN` | Redirect an assigned unit to a different incident | Unit must be `DISPATCHED`, `ON_SCENE`, or `TRANSPORTING` |
| `STAGE` | Pre-position a unit near an incident without committing | Unit must be `AVAILABLE`; incident must be `PENDING` |
| `MUTUAL_AID` | Request external unit of a given type | Only allowed when all local units of that type are busy |
| `UPGRADE` | Increase incident severity | New severity must be strictly higher than current |
| `DOWNGRADE` | Decrease incident severity | New severity must be strictly lower than current |

---

## Observation Space

**`src.models.Observation`**

| Field | Type | Description |
|---|---|---|
| `result` | `str` | Human-readable result of the last action |
| `score` | `float` | Episode score in `[0.0, 1.0]` (task-level grade) |
| `protocol_ok` | `bool` | Whether the action passed protocol validation |
| `issues` | `list[str]` | Warnings or error codes from the validator |
| `reward_breakdown` | `dict[str, float] \| None` | Per-component reward scores for dashboard display |

**Full State (`src.models.State`)**

| Field | Type | Description |
|---|---|---|
| `units` | `dict[str, UnitState]` | All units with type, status, location, ETA |
| `incidents` | `dict[str, IncidentState]` | All incidents with type, severity, status, assigned units |
| `episode_id` | `str` | Unique episode identifier |
| `step_count` | `int` | Current step number |
| `task_id` | `str` | Active task identifier |
| `city_time` | `float` | Simulated city clock in seconds (30s per step) |
| `metadata` | `dict` | Schema info, districts, seeds, wave configs, bookkeeping |

**Unit Status Transitions**

```
AVAILABLE → DISPATCHED → ON_SCENE → AVAILABLE
                ↓
         OUT_OF_SERVICE (shift_surge only)
```

**Incident Status Transitions**

```
PENDING → RESPONDING → ON_SCENE → RESOLVED
   ↓           ↓
ESCALATED   ESCALATED    (survival clock expires)
```

---

## Reward Function

The step-level reward is a weighted combination of five components:

| Component | Weight | Description |
|---|---|---|
| `response_time` | **30%** | How quickly dispatched units reach incidents relative to severity benchmarks (P1: 240s, P2: 480s, P3: 900s) |
| `triage` | **25%** | Whether the dispatched unit type matches incident requirements (e.g., MEDIC for CARDIAC_ARREST) |
| `survival` | **25%** | Fraction of Priority-1 incidents resolved before the survival clock expires |
| `coverage` | **12%** | Geographic distribution of available units across city districts |
| `protocol` | **8%** | Action legality + optional phraseology/readback quality via `Action.notes` |

**Safety Gate**: If any Priority-1 incident was seen and the survival score is `0.0`, the total reward is hard-capped at `0.2` regardless of efficiency gains.

**Non-DISPATCH actions** receive neutral `0.5` for `response_time` and `triage`, allowing agents to maintain coverage without penalty.

---

## Tasks

### Task Difficulty Overview

| Task | Difficulty | Max Steps | Key Challenge |
|---|---|---|---|
| `single_incident` | 🟢 Easy | 20 | Dispatch the right unit type quickly |
| `multi_incident` | 🟡 Medium | 40 | Triage 3 simultaneous incidents, protect P1s |
| `mass_casualty` | 🔴 Hard | 60 | Manage wave-based surge with limited resources |
| `shift_surge` | 🔴 Hard | 60 | Adapt as units fail and incidents stream continuously |

---

### 🟢 Task 1: `single_incident` — Basic Dispatch (Easy)

**Scenario**: One active incident (`CARDIAC_ARREST`, Priority-1) in a small city. A MEDIC, ENGINE, and PATROL are all available.

**Objective**: Dispatch the correct unit type (MEDIC) to the incident as fast as possible.

**Grader Logic**:
```
score = 0.0
if incident RESOLVED:          score += 0.50
if MEDIC dispatched correctly: score += 0.30
if resolved within 10 steps:   score += 0.20
```

**Why it's easy**: One incident, one correct action, small state space.

**What a good agent does**: Immediately dispatches `MED-1 → INC-001`.

---

### 🟡 Task 2: `multi_incident` — Simultaneous Triage (Medium)

**Scenario**: Three concurrent incidents at episode start — a structure fire (P2), a cardiac arrest (P1), and a shooting (P1) — with 6 available units.

**Objective**: Respond to all incidents with the right unit types, prioritizing P1s.

**Grader Logic**:
```
score = 0.5 × p1_resolution_rate
      + 0.3 × overall_resolution_rate
      - 0.2 × escalation_penalty
```

**Why it's medium**: Multiple incidents compete for units; wrong type dispatch wastes coverage; P1s must be addressed before P2.

**What a good agent does**: Immediately dispatches MEDIC to cardiac arrest and patrol to shooting, then handles the fire with ENGINE/LADDER.

---

### 🔴 Task 3: `mass_casualty` — Wave-Based Surge (Hard)

**Scenario**: One critical incident (`BUILDING_COLLAPSE`, P1) at step 0. New waves arrive at steps 5 (structure fire) and 12 (two simultaneous cardiac arrests).

**Objective**: Maximize P1 survival across all waves despite resource conflicts.

**Grader Logic**:
```
score = 0.6 × p1_survival_rate
      + 0.3 × mean_step_reward
      - failure_penalty
```

**Why it's hard**: Resources are exhausted when waves arrive. Agents must decide whether to reassign mid-scene or request mutual aid (at a 120s ETA penalty). Mutual aid is only legal when local units of the required type are fully committed.

**What a good agent does**: Dispatches immediately to initial collapse, stages additional units near expected wave arrival zones, requests mutual aid for later waves.

---

### 🔴 Task 4: `shift_surge` — Long-Horizon Degradation (Hard)

**Scenario**: 5 units start available, but 3 go `OUT_OF_SERVICE` by step 5. Incidents arrive in waves every 8 steps throughout the 60-step episode.

**Objective**: Maintain city-wide throughput and P1 survival despite progressive resource degradation.

**Grader Logic**:
```
score = 0.35 × resolution_ratio
      + 0.25 × p1_survival
      + 0.15 × coverage
      + 0.15 × (1 - backlog_ratio)
      + 0.10 × mean_reward
      - 0.25 × escalation_ratio
```

**Why it's hard**: No single optimal strategy — agents must continuously rebalance between throughput and coverage as available resources shrink and incident demand grows.

---

## Unit Types

| Unit | Code | Speed | Primary Use |
|---|---|---|---|
| Engine | `ENGINE` | 0.8 bl/s | Structure fires, hazmat support |
| Ladder | `LADDER` | 0.6 bl/s | Multi-story fires, rescues |
| Medic | `MEDIC` | 1.0 bl/s | Medical emergencies, trauma |
| Patrol | `PATROL` | 1.2 bl/s | Shootings, MVAs, crowd control |
| Hazmat | `HAZMAT` | 0.5 bl/s | Chemical/biological spills |

## Incident Types

| Incident | Recommended Units | Default Severity |
|---|---|---|
| `CARDIAC_ARREST` | MEDIC | P1 |
| `STRUCTURE_FIRE` | ENGINE × 2, LADDER | P2 |
| `SHOOTING` | MEDIC, PATROL × 2 | P1 |
| `MULTI_VEHICLE_ACCIDENT` | MEDIC, PATROL | P2 |
| `BUILDING_COLLAPSE` | ENGINE, LADDER, MEDIC × 2 | P1 |
| `HAZMAT_SPILL` | HAZMAT, ENGINE | P2 |
| `OVERDOSE` | MEDIC | P2 |
| `MISSING_PERSON` | PATROL | P3 |

---

## OpenEnv Interface

```python
import asyncio
from src.openenv_environment import OpenEnvEnvironment
from src.models import Action, DispatchAction

async def main():
    env = OpenEnvEnvironment(task_id="multi_incident", seed=42)

    # Reset to initial state
    obs = await env.reset()
    print(obs.result)  # "dispatch center online"

    # Get legal actions (protocol-validated)
    legal = env.legal_actions()

    # Take a step
    action = legal[0]
    obs, reward, done = await env.step(action)

    print(f"reward={reward:.3f}, done={done}, protocol_ok={obs.protocol_ok}")

    # Inspect full state
    state = env.state()
    print(f"step={state.step_count}, city_time={state.city_time}s")

asyncio.run(main())
```

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "healthy"}` |
| `/reset` | POST | Reset environment; body: `{"task_id": "...", "seed": 42}` (both optional) |
| `/step` | POST | Execute an action; body: `{"action": {...}}` |
| `/state` | GET | Current full environment state |
| `/tasks` | GET | List all available tasks with metadata |
| `/dashboard/state` | GET | Extended state for live HTML dashboard |
| `/schema` | GET | JSON schemas for Action, Observation, State |
| `/metadata` | GET | Environment name, version, description |

---

## Quick Start

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Run the demo (non-interactive, no LLM required)
uv run python demo.py

# Start the API server
uv run python -m src.server.app

# Run inference with random baseline (no API key required)
USE_RANDOM=true \
  API_BASE_URL=https://api.openai.com/v1 \
  MODEL_NAME=gpt-4 \
  OPENAI_API_KEY=dummy \
  uv run python inference.py

# Run full test suite
uv run pytest tests/ -v
```

### Using pip

```bash
pip install -r requirements.txt
python demo.py
```

---

## Docker

### Build & Run

```bash
# Build image
docker build -t citywide-dispatch-supervisor .

# Run (defaults to port 8000)
docker run -p 8000:8000 citywide-dispatch-supervisor

# Run on custom port (for HF Spaces)
docker run -e PORT=7860 -p 7860:7860 citywide-dispatch-supervisor

# Health check
curl http://localhost:8000/health

# Reset to a specific task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "multi_incident", "seed": 42}'
```

---

## Hugging Face Spaces Deployment

This repository is deployed as a Docker-based HF Space.

1. Create a new HF Space → select **Docker**
2. Push this repository to the Space
3. The server reads `PORT` from the environment (HF sets `PORT=7860`)
4. Once running, the following endpoints are publicly available:
   - `GET /health`
   - `POST /reset`
   - `POST /step`
   - `GET /state`

Validate your deployment with the prevalidation script:

```bash
bash samplematerial/prevalidation.sh https://your-space.hf.space .
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint base URL |
| `MODEL_NAME` | Yes | Model identifier string |
| `OPENAI_API_KEY` | Yes (unless `USE_RANDOM=true`) | API key for the OpenAI client |
| `USE_RANDOM` | No | Set to `true` to run a deterministic random baseline agent (no API key needed) |
| `PORT` | No | Server port (default: 8000; HF Spaces sets this automatically) |

> **Backwards compatibility**: `HF_TOKEN` is accepted as a fallback for `OPENAI_API_KEY`.

---

## Baseline Scores

Run the random baseline agent against all tasks:

```bash
USE_RANDOM=true \
  API_BASE_URL=https://api.openai.com/v1 \
  MODEL_NAME=gpt-4 \
  OPENAI_API_KEY=dummy \
  uv run python inference.py
```

| Task | Difficulty | Random Baseline Score |
|---|---|---|
| `single_incident` | Easy | ~0.55 |
| `multi_incident` | Medium | ~0.48 |
| `mass_casualty` | Hard | ~0.32 |
| `shift_surge` | Hard | ~0.38 |

*Scores use `seed=42` for reproducibility. Variance is low across runs due to deterministic state machine.*

---

## Project Structure

```
.
├── src/
│   ├── models.py               # Pydantic typed contracts (Action, Observation, State)
│   ├── protocol.py             # Dispatch protocol validator
│   ├── physics.py              # City-grid movement / ETA helpers
│   ├── city_schema.py          # City topology + unit configuration loader
│   ├── state_machine.py        # Core dispatch state machine
│   ├── rewards.py              # Reward engine + episode graders
│   ├── phraseology.py          # Dispatch phraseology renderer/judge
│   ├── api.py                  # REST API client wrapper
│   ├── grading.py              # Centralized episode grading router
│   ├── benchmark.py            # Benchmark runner (list/run all tasks)
│   ├── openenv_environment.py  # OpenEnv-compatible environment wrapper
│   ├── tasks/
│   │   ├── registry.py         # Task registry + deterministic scenario fixtures
│   │   ├── single_incident.py  # Easy task + grader
│   │   ├── multi_incident.py   # Medium task + grader
│   │   ├── mass_casualty.py    # Hard task + grader
│   │   └── shift_surge.py      # Hard task + grader
│   ├── server/
│   │   ├── app.py              # FastAPI server (reset/step/state endpoints)
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   └── visualizer/
│       └── viewer.py           # Read-only 2D Matplotlib visualizer
├── data/
│   ├── metro_city.json         # Large city schema (default)
│   └── city_small.json         # Small city schema (testing)
├── tests/                      # TDD test suite (~20 test modules)
├── samplematerial/
│   └── prevalidation.sh        # HF Space + Docker validation script
├── demo.py                     # Non-interactive demo (no LLM required)
├── inference.py                # Competition inference script
├── live_dashboard.html         # Browser-based live dashboard
├── validate_local.py           # Local pre-submission validation
├── openenv.yaml                # OpenEnv specification
├── pyproject.toml              # uv project config
├── requirements.txt            # pip dependencies
└── Dockerfile                  # Root Docker build
```

---

## Live Dashboard

After starting the server and calling `/reset`, open `live_dashboard.html` in a browser:

```bash
# Terminal 1: start server
uv run python -m src.server.app

# Terminal 2: reset to a task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "multi_incident"}'

# Browser: open live_dashboard.html
```

The dashboard polls `/dashboard/state` every 500ms and renders:

- Unit cards (status, ETA, assignment, location)
- Incident cards (type, severity, status, assigned units)
- City map (2D grid with unit and incident markers)
- Per-step reward component bars

---

## 2D Visualizer (Programmatic)

```python
import asyncio
from src.openenv_environment import OpenEnvEnvironment
from src.visualizer.viewer import Viewer2D

async def main():
    env = OpenEnvEnvironment(task_id="multi_incident", seed=42)
    await env.reset()
    Viewer2D().render_to_file("frame.png", env.state())
    env.close()

asyncio.run(main())
```

---

## Determinism

All scenarios are deterministic under a fixed seed:

```python
env1 = OpenEnvEnvironment(task_id="shift_surge", seed=42)
env2 = OpenEnvEnvironment(task_id="shift_surge", seed=42)
# env1 and env2 produce identical episodes
```

Incident positions include small seeded perturbations for realism; the overall episode structure (waves, unit positions, incident types) is fully reproducible.

---

## Running Tests

```bash
# Full test suite
uv run pytest tests/ -v

# Individual modules
uv run pytest tests/test_state_machine.py -v
uv run pytest tests/test_rewards.py -v
uv run pytest tests/test_openenv_integration.py -v
uv run pytest tests/test_inference.py -v
```

---

## Pre-Submission Validation

```bash
# Full local validation (tests + inference + docker + benchmark scores)
uv run python validate_local.py

# OpenEnv spec validation
uv run openenv validate

# HF Space validation (requires deployed space)
bash samplematerial/prevalidation.sh https://your-space.hf.space .
```

---

## License

MIT License