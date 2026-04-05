# 911 City-Wide Emergency Dispatch Supervisor

**LLM-powered airport ground control — full lifecycle**

A unified RL training environment for airport ground control operations, powered by LLM orchestration. Manages the complete aircraft lifecycle from approach to departure, including landing, handoff, taxiing, docking, pushback, and takeoff sequencing.

## Overview

This project implements a competition-ready benchmark environment for training and evaluating LLM agents in airport ground control operations. It features:

- **Full lifecycle coverage**: 11 distinct phases from approach to departure
- **Deterministic simulation**: Reproducible episodes under fixed seeds
- **Normalized hybrid ATC protocol**: Airport-agnostic communication procedures
- **OpenEnv compatible**: Standard RL environment interface
- **Read-only 2D visualizer**: Synchronized state visualization

## Tasks

### 1. Arrival Task (Hard)

Complete inbound aircraft lifecycle:
- Landing clearance and runway operations
- Tower-to-ground frequency handoff
- Taxi-in to assigned gate
- Safe docking

### 2. Departure Task (Easy)

Complete outbound aircraft lifecycle:
- Pushback approval from gate
- Taxi-out via taxiway network
- Departure queue management
- Runway release and takeoff clearance

### 3. Integrated Task (Medium)

Full turnaround lifecycle:
- Complete arrival sequence (landing → docking)
- Turnaround operations
- Complete departure sequence (pushback → departure)

## Quick Start

### Using uv (Recommended)

```bash
# Install dependencies
uv sync

# Run the demo (non-interactive episode visualization)
uv run python demo.py

# Run inference with LLM agent
uv run python inference.py
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

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── models.py           # Pydantic typed contracts
│   ├── protocol.py         # Normalized ATC protocol
│   ├── physics.py          # Deterministic aircraft physics
│   ├── airport_schema.py   # Airport topology
│   ├── state_machine.py    # Full lifecycle state machine
│   ├── rewards.py          # Reward engine and graders
│   ├── phraseology.py      # ATC phraseology judge
│   ├── api.py              # REST API surface
│   ├── openenv_environment.py  # OpenEnv wrapper
│   ├── tasks/
│   │   ├── registry.py     # Task registry
│   │   ├── arrival.py      # Arrival task implementation
│   │   ├── departure.py    # Departure task implementation
│   │   └── integrated.py  # Integrated task implementation
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
docker build -t gradient-ascent-atc .
```

### Run

```bash
# Run container
docker run -p 8000:8000 gradient-ascent-atc

# Health check
curl http://localhost:8000/health

# Reset environment
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{"task_id": "arrival", "seed": 42}'
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | OpenAI API base URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model to use | `gpt-4` |
| `HF_TOKEN` | HuggingFace token | None |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment to initial state |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current environment state |

## HF Space

**Placeholder**: [https://huggingface.co/spaces/gradient-ascent/atc-ground-control](https://huggingface.co/spaces/gradient-ascent/atc-ground-control)

## License

MIT License
