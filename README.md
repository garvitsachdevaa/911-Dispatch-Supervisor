---
title: 911 Dispatch Supervisor
emoji: 🚨
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

# 911 City-Wide Emergency Dispatch Supervisor

**LLM-powered 911 dispatch supervision — city scale**

A unified RL training environment for city-wide emergency dispatch operations. The agent supervises police, fire, and EMS unit allocation across simultaneous incidents under a deterministic simulation.

## Overview

This project implements a benchmark environment for training and evaluating LLM agents as emergency dispatch supervisors. It features:

- **Dispatch lifecycle**: incidents advance from pending to resolved (or escalated)
- **Deterministic simulation**: Reproducible episodes under fixed seeds
- **Protocol validator**: Checks if actions are legal in the current state
- **OpenEnv compatible**: Standard RL environment interface
- **Read-only 2D visualization**: Synchronized unit/incident visualization

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | OpenAI-compatible endpoint base URL |
| `MODEL_NAME` | Yes | Model identifier string |
| `HF_TOKEN` | Yes (unless `USE_RANDOM=true`) | API key / HF token |
| `USE_RANDOM` | No | Set to `true` to use deterministic random agent (no LLM) |

## Tasks

### 1. `single_incident`

One active incident and a small unit pool. Focus: basic dispatch, response-time, and protocol correctness.

### 2. `multi_incident`

Multiple concurrent incidents with limited resources. Focus: triage and prioritization.

### 3. `mass_casualty`

High severity surge (Priority-1 heavy). Focus: survival outcomes and rapid allocation.

### 4. `shift_surge`

Longer horizon with incident waves and unit availability changes. Focus: coverage and strategic staging.

### Task Difficulty Guide

| Task | Difficulty | Key Challenge | Success Criteria |
|------|-----------|---------------|-----------------|
| `single_incident` | Easy | Dispatch the right unit type (MEDIC) quickly | Incident resolved, correct unit, ETA < 300s |
| `multi_incident` | Medium | Triage 3 simultaneous incidents, prioritize P1 | All P1 incidents responded to, no ESCALATED |
| `mass_casualty` | Hard | Manage wave-based surge with limited resources | Maximize P1 survival rate across waves |
| `shift_surge` | Hard | Adapt as units go out of service over time | Maintain coverage and resolve incidents despite attrition |

## Contracts

### Action

`src.models.Action` fields:

| Field | Type | Notes |
|------|------|-------|
| `action_type` | `DispatchAction` | e.g. `DISPATCH`, `CANCEL`, `UPGRADE`, `MUTUAL_AID` |
| `unit_id` | `str` | Unit identifier, e.g. `MED-1` |
| `incident_id` | `str` | Incident identifier, e.g. `INC-0001` |
| `notes` | `str \| None` | Optional free text |
| `priority_override` | `IncidentSeverity \| None` | Required for upgrade/downgrade |

### Observation

`src.models.Observation` fields:

| Field | Type | Notes |
|------|------|-------|
| `result` | `str` | Human-readable result |
| `score` | `float` | Step reward in `[0,1]` |
| `protocol_ok` | `bool` | Whether action was legal |
| `issues` | `list[str]` | Warnings/errors from protocol validator |
| `reward_breakdown` | `dict[str,float] \| None` | Component scores for dashboard |

## Reward

The reward is a weighted combination of:

- `response_time`
- `triage`
- `survival`
- `coverage`
- `protocol`

See `src/rewards.py` for the component definitions and weights.

## Quick Start

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Run the demo (non-interactive episode visualization)
uv run python demo.py

# Run inference with LLM agent
uv run python inference.py

# Run API server
uv run python -m src.server.app

# Open live dashboard (static HTML)
# - start the server first
# - call /reset to initialize the environment
# - then open live_dashboard.html in a browser
```

### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py

# Run inference
python inference.py
```

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

## Reward Function

The reward signal is a weighted combination of five components:

| Component | Weight | Description |
|-----------|--------|-------------|
| `response_time` | 30% | How quickly units reach incidents relative to severity benchmarks |
| `triage` | 25% | Whether the dispatched unit type matches incident requirements |
| `survival` | 25% | Whether Priority-1 incidents are resolved before survival clock expires |
| `coverage` | 12% | Geographic distribution of available units across city districts |
| `protocol` | 8% | Whether the dispatch action was legally valid |

**Safety gate:** If any Priority-1 incident was seen and `survival=0.0`, the total episode score is capped at `0.2` regardless of other components.



## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── models.py           # Pydantic typed contracts
│   ├── protocol.py         # Dispatch protocol validator
│   ├── physics.py          # City-grid movement/ETA helpers
│   ├── city_schema.py      # City topology + unit configuration
│   ├── state_machine.py    # Dispatch state machine
│   ├── rewards.py          # Reward engine and graders
│   ├── phraseology.py      # Dispatch phraseology judge
│   ├── api.py              # REST API surface
│   ├── openenv_environment.py  # OpenEnv wrapper
│   ├── tasks/
│   │   ├── registry.py     # Task registry
│   │   ├── single_incident.py
│   │   ├── multi_incident.py
│   │   ├── mass_casualty.py
│   │   └── shift_surge.py
│   ├── server/
│   │   ├── app.py          # FastAPI server
│   │   └── Dockerfile
│   └── visualizer/
│       └── viewer.py       # 2D visualization
├── tests/                  # TDD test suite
├── demo.py                 # Demo script
├── inference.py            # LLM inference script
├── requirements.txt        # pip dependencies
├── pyproject.toml          # uv project config
├── openenv.yaml           # OpenEnv specification
└── README.md
```

## Docker Deployment

### Build

```bash
docker build -t citywide-dispatch-supervisor .
```

### Run

```bash
# Run container
docker run -p 8000:8000 citywide-dispatch-supervisor

# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "single_incident", "seed": 42}'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment to initial state |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current environment state |
| `/dashboard/state` | GET | Extended state for `live_dashboard.html` |
| `/tasks` | GET | List all available tasks with metadata |

## HF Space

**Placeholder**: (add link here)

## License

MIT License
