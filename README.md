---
title: 911 Dispatch Supervisor
emoji: "🚨"
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# 🚨 911 Dispatch Supervisor

> **A city-wide emergency dispatch RL environment** — train and evaluate LLM agents to manage simultaneous incidents by dispatching police, fire, and EMS units across a city grid under realistic resource constraints.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://openenv.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://hub.docker.com)
[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## Why This Matters

911 dispatch centers in the United States handle over 240 million calls per year. Every dispatcher decision — which unit to send, in what order, with what priority — directly determines survival outcomes. A 90-second delay in dispatching a MEDIC to a cardiac arrest drops survival probability by roughly 10%.

The **911 Dispatch Supervisor** is the first open RL benchmark for training and evaluating AI agents on emergency dispatch decisions. It models the exact tradeoffs real dispatchers face: triage under uncertainty, multi-unit resource allocation, geographic coverage, and protocol compliance — all simultaneously.

This fills a direct gap for researchers building AI copilots for public safety systems, and provides immediate evaluation value for any LLM claiming real-world decision-making capability.

## Overview

At every step, an LLM agent plays the role of a city-wide dispatch supervisor, deciding which units to dispatch, reassign, cancel, stage, or escalate — under time pressure, limited resources, and competing priorities across a 100×100 city grid.

This is not a toy environment. Emergency dispatch is a high-stakes, multi-objective decision problem that:
- Requires **triage** — prioritizing life-threatening incidents over property damage
- Demands **coverage awareness** — keeping geographic zones protected
- Rewards **correct unit-type matching** — sending a MEDIC vs. an ENGINE
- Punishes **delays** that cause Priority-1 incidents to escalate
- Scores **dispatch phraseology** — realistic radio communication language

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

#### Dispatch Phraseology (bonus scoring)

The `notes` field is scored for realistic radio communication language. Agents that use proper dispatch phraseology receive up to 8% bonus on their protocol score.

| Action | Example notes value |
|---|---|
| Dispatch MEDIC to cardiac | `"Medic 1 en route to cardiac arrest, Code 3, ETA 4 minutes"` |
| Dispatch ENGINE to fire | `"Engine 2 responding to structure fire, Code 3, all units advised"` |
| Mutual aid request | `"Requesting mutual aid, all local MEDICs committed, Priority 1 cardiac at grid 45-72"` |
| Stage unit | `"Engine 1 staging at District 3 perimeter, awaiting scene clear"` |

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

> **⚠️ Safety Gate:** If any Priority-1 incident (cardiac arrest, shooting, building collapse) results in zero survival score, the entire episode reward is hard-capped at **0.2** regardless of other performance. This forces agents to treat life-threatening incidents as non-negotiable — exactly as real dispatch protocol requires.

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

**Scoring:** 50% resolution + 30% correct unit type used + 20% response speed.

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

**Scoring:** 50% P1 resolution + 30% overall resolution − 20% escalation penalty.

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

**Scoring:** 60% P1 survival + 30% mean step reward − failure penalty if building collapse unresponded.

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

**Scoring:** 35% resolution + 25% P1 survival + 15% coverage + 15% backlog management + 10% step reward − 25% escalation penalty.

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

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (non-interactive, no LLM required)
python demo.py

# Start the API server
python -m src.server.app

# Run random agent baseline (no API key required)
USE_RANDOM=true python inference.py

# Run LLM agent
API_BASE_URL=https://router.huggingface.co/v1 \
  MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  HF_TOKEN=your_token \
  python inference.py

# Run full test suite
pytest tests/ -v
```

---

## Docker

### Build & Run

```bash
# Build image
docker build -t citywide-dispatch-supervisor .

# Run on port 7860 (required for HF Spaces)
docker run -p 7860:7860 citywide-dispatch-supervisor

# Health check
curl http://localhost:7860/health

# Reset to a specific task
curl -X POST http://localhost:7860/reset \
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

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | Model identifier | `meta-llama/Llama-3.1-8B-Instruct` |
| `HF_TOKEN` | HuggingFace API key | — |
| `USE_RANDOM` | Set `true` for deterministic random baseline | `false` |
| `PORT` | Server port | `7860` |

---

## Baseline Scores

Scores normalized to `[0.0, 1.0]` using `sum(rewards) / max_steps`.  
Run with `USE_RANDOM=true python inference.py` (seed=42, fully deterministic).

| Task | Difficulty | Max Steps | Random Agent Score |
|---|---|---|---|
| `single_incident` | Easy | 20 | 0.2000 |
| `multi_incident` | Medium | 40 | 0.3117 |
| `mass_casualty` | Hard | 60 | 0.4645 |
| `shift_surge` | Hard | 60 | 0.3183 |

> **Note:** Earlier README versions showed higher scores (~0.30–0.74) from a different scoring path (`observation.score`). These figures use the canonical competition normalization: `sum(step_rewards) / max_steps`, clamped to `[0.0, 1.0]`.

### What the scores mean

A random agent scoring **0.20 on the easiest task** confirms the environment is not trivially solvable — there is no reward for random dispatching. The gradient from 0.20 → 0.46 across tasks reflects genuine increasing complexity, not just more steps.

A well-prompted frontier LLM (GPT-4o, Llama-3.1-70B) is expected to score **0.55–0.75 on single_incident** and **0.30–0.45 on shift_surge**, demonstrating the environment meaningfully differentiates agent capability.

LLM agents (`meta-llama/Llama-3.1-8B-Instruct` via `https://router.huggingface.co/v1`) are expected to score meaningfully higher on easy and medium tasks by correctly prioritizing P1 incidents and matching unit types.

Run the baseline matrix (random + LLM reruns) and emit a JSON report:

```bash
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
HF_TOKEN=your_token \
python scripts/run_baseline_matrix.py --random-runs 1 --llm-runs 3 --output-json baseline_report.json
```

Windows PowerShell shortcut:

```powershell
$env:HF_TOKEN="your_token"
powershell -ExecutionPolicy Bypass -File scripts/run_nemotron_baseline.ps1 -RandomRuns 1 -LlmRuns 3
```

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
python -m src.server.app

# Terminal 2: reset to a task
curl -X POST http://localhost:7860/reset \
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
pytest tests/ -v

# Individual modules
pytest tests/test_state_machine.py -v
pytest tests/test_rewards.py -v
pytest tests/test_openenv_integration.py -v
pytest tests/test_inference.py -v
```

---

## Pre-Submission Validation

```bash
# Full local validation (tests + inference + docker + benchmark scores)
python validate_local.py

# OpenEnv spec validation
openenv validate

# HF Space validation (requires deployed space)
bash samplematerial/prevalidation.sh https://your-space.hf.space .

# Windows (explicit Git Bash)
"C:/Program Files/Git/bin/bash.exe" samplematerial/prevalidation.sh https://your-space.hf.space .
```

---

## License

MIT License
