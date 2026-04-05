# ATC Ground Control RL Environment — Full Lifecycle

## TL;DR

> **Quick Summary**: Build a unified RL training environment where an LLM agent learns to manage full airport ground-control operations — landing, arrival handoff, taxi-in, docking, pushback, taxi-out, and departure — using a normalized hybrid ATC protocol. Includes a read-only 2D visualizer synchronized to environment state, REST/API-driven interaction, and OpenEnv competition compliance.
>
> **Deliverables**:
> - Core RL environment with deterministic aircraft and full lifecycle state machine
> - Normalized hybrid ATC protocol layer (airport-agnostic)
> - REST/API endpoint surface for agent interaction
> - Read-only 2D visualizer synchronized to environment state
> - OpenEnv-compliant packaging: `openenv.yaml`, typed models, `reset()`/`step()`/`state()`, Docker, root `inference.py`
> - Competition-ready: 3+ graded tasks, strict stdout logging, <20min runtime, 2vCPU/8GB compatible
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: T1 → T2 → T5 → T7 → T9 → T11 → T14 → T17 → F1-F4

---

## Context

### Original Request
Train an RL/LLM regime to manage airport ground control, specifically the multitasking bottleneck of human communication and logistics. The system should handle handover/handoff, air/ground frequency management, docking, undocking, clearance, taxiing, takeoffs, and landing procedures. The agent should be an effective task manager and standardized communication medium.

### Revised Scope
- **Full lifecycle**: landing → arrival handoff → taxi-in → docking → pushback → taxi-out → departure sequencing
- **Unified RL environment**: one environment for training and demonstration, not separate tracks
- **Normalized hybrid protocol**: airport-agnostic ATC procedures, not FAA/CAA-locked
- **Read-only 2D visualizer**: synchronized viewer showing aircraft positions, runways, taxiways, callsign/heading/altitude
- **Realistic-enough physics**: glide path, descent rate, heading/altitude tracking — no flap/minutiae simulation
- **REST/API-driven**: environment exposes endpoints for agent interaction
- **Deterministic aircraft**: environment-side actors; LLM is the orchestrator
- **Clean repo structure**: organized, submission-ready layout with no reference-repo clutter

### Competition Constraints (from hackathon spec + sample files)
- HF Space must return 200 on `POST /reset` (validated by `prevalidation.sh`)
- `openenv.yaml` + typed Pydantic models (`Action`, `Observation`, `State`)
- `step()`/`reset()`/`state()` endpoints — **async interface** (`await env.reset()`, `await env.step()`)
- Dockerfile must build from root or `server/` directory (validated by `prevalidation.sh`)
- Root `inference.py` using **OpenAI client** with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
- **Exact stdout format**:
  - `[START] task=<task_name> env=<benchmark> model=<model_name>`
  - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
  - `[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>`
- 3+ tasks with graders producing scores in `[0.0, 1.0]`
- Runtime <20 minutes on 2 vCPU / 8 GB RAM
- Env loadable via `from_docker_image(IMAGE_NAME)` pattern
- `demo.py` script for HF Space landing page
- `requirements.txt` at repo root

---

## Work Objectives

### Core Objective
Deliver a unified RL training environment for airport ground-control management that is competition-ready, visually demonstrable, and architecturally extensible to any airport.

### Concrete Deliverables
- `openenv.yaml` and environment package scaffold
- Strict Pydantic typed models for Action, Observation, State, and protocol types
- Full-lifecycle airport state machine (landing through departure)
- Deterministic aircraft simulation with realistic-enough physics
- Normalized hybrid ATC protocol layer
- REST/API endpoint surface for agent interaction
- Read-only 2D visualizer synchronized to environment state
- Root `inference.py` with exact competition logging format
- Docker/HF packaging and validation path
- 3+ graded tasks with normalized scores

### Definition of Done
- [ ] `openenv validate` passes
- [ ] Docker build succeeds
- [ ] `python inference.py` completes with correct stdout format
- [ ] All graded tasks return scores in `[0.0, 1.0]`
- [ ] 2D visualizer synchronizes to environment state
- [ ] Full lifecycle works: landing → arrival handoff → taxi-in → docking → pushback → taxi-out → departure
- [ ] TDD tests pass across all modules

### Must Have
- Full operational lifecycle coverage
- Deterministic aircraft as environment actors
- LLM/RL agent as orchestrator via REST/API
- Normalized hybrid ATC protocol
- Read-only 2D visualizer
- OpenEnv compliance
- Competition-ready packaging
- TDD workflow
- Strict Pydantic typing

### Must NOT Have (Guardrails)
- No low-level aircraft systems (flap settings, engine minutiae)
- No interactive visualizer controls (read-only only)
- No FAA/CAA-locked protocol (must be airport-agnostic)
- No scope creep into en-route/TRACON control
- No live airport-data fetches at runtime
- No reference-repo dependencies in the submission

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed.

### Test Decision
- **Infrastructure exists**: NO — will be created during TDD
- **Automated tests**: TDD (RED → GREEN → REFACTOR)
- **Framework**: `pytest` or `unittest` (via `uv run python`)
- **Package management**: `uv` for all Python environment operations (init, add, run, sync)
- **If TDD**: Each task includes failing tests before implementation

### QA Policy
Every task MUST include agent-executed QA scenarios. Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **API/Backend**: Bash (`curl`) or Python test runner
- **CLI/Inference**: Bash execution with stdout capture
- **Schema/Typing**: Test runner and import-level contract tests
- **Docker/HF**: Docker build + container run + endpoint pings
- **Visualizer**: Screenshot capture + state synchronization checks

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — foundations + contracts):
├── T1: Repo restructuring + clean package layout [quick]
├── T2: Strict typed Pydantic models (extended for full lifecycle) [quick]
├── T3: Normalized hybrid ATC protocol definitions [deep]
├── T4: Deterministic aircraft physics model (glide path, descent, surface) [deep]
├── T5: Airport schema layer (Gatwick-first, extensible) [quick]
└── T6: TDD/test harness + validator smoke scaffolding [quick]

Wave 2 (After Wave 1 — core environment engine):
├── T7: Full-lifecycle state machine (landing → departure) [deep]
├── T8: REST/API endpoint surface for agent interaction [unspecified-high]
├── T9: Reward engine + grader primitives [unspecified-high]
├── T10: Task registry + deterministic scenario fixtures [quick]
├── T11: OpenEnv server/client integration (reset/step/state) [quick]
└── T12: Phraseology renderer/judge (normalized subset) [unspecified-high]

Wave 3 (After Wave 2 — task implementations + visualizer):
├── T13: Arrival task (landing + handoff + taxi-in + docking) [deep]
├── T14: Departure task (pushback + taxi-out + release sequencing) [deep]
├── T15: Integrated full-lifecycle task (end-to-end episode) [deep]
├── T16: Read-only 2D visualizer synchronized to env state [visual-engineering]
└── T17: Root inference.py with exact stdout format + OpenAI client [quick]

Wave 4 (After Wave 3 — packaging + submission):
├── T18: Demo script + HF Space landing page + requirements.txt [quick]
├── T19: Docker/HF Space packaging + health/reset validation [quick]
├── T20: Runtime/resource hardening + determinism [unspecified-high]
└── T21: Benchmark assembly + prevalidation.sh integration [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── F1: Plan compliance audit (oracle)
├── F2: Code quality review (unspecified-high)
├── F3: Real manual QA (unspecified-high)
└── F4: Scope fidelity check (deep)

Critical Path: T1 → T2 → T5 → T7 → T9 → T11 → T14 → T17 → T20 → F1-F4
Parallel Speedup: high (6 concurrent in Wave 1, 6 in Wave 2, 5 in Wave 3)
Max Concurrent: 6
```

### Dependency Matrix

- **T1**: Blocks — T2, T3, T4, T5, T6, T17, T18, T20
- **T2**: Blocks — T7, T8, T9, T10, T11, T12, T13, T14, T15, T16
- **T3**: Blocks — T7, T12, T13, T14, T15
- **T4**: Blocks — T7, T13, T14, T15, T16, T19
- **T5**: Blocks — T7, T10, T13, T14, T15, T16, T20
- **T6**: Blocks — T8, T9, T10, T11, T12, T13, T14, T15, T17, T18, T19, T20
- **T7**: Blocks — T8, T9, T10, T11, T13, T14, T15, T19
- **T8**: Blocks — T11, T16, T17, T18
- **T9**: Blocks — T10, T13, T14, T15, T20
- **T10**: Blocks — T13, T14, T15, T19, T20
- **T11**: Blocks — T16, T17, T18, T20
- **T12**: Blocks — T13, T14, T15, T17
- **T13**: Blocks — T15, T20
- **T14**: Blocks — T15, T17, T20
- **T15**: Blocks — T17, T20
- **T16**: Blocks — T17
- **T17**: Blocks — T18, T20
- **T18**: Blocks — T20
- **T19**: Blocks — T20
- **T20**: Blocks — F1, F2, F3, F4

### Agent Dispatch Summary

- **Wave 1**: 6 agents — T1 `quick`, T2 `quick`, T3 `deep`, T4 `deep`, T5 `quick`, T6 `quick`
- **Wave 2**: 6 agents — T7 `deep`, T8 `unspecified-high`, T9 `unspecified-high`, T10 `quick`, T11 `quick`, T12 `unspecified-high`
- **Wave 3**: 5 agents — T13 `deep`, T14 `deep`, T15 `deep`, T16 `visual-engineering`, T17 `quick`
- **Wave 4**: 4 agents — T18 `quick`, T19 `quick`, T20 `unspecified-high`, T21 `unspecified-high`
- **FINAL**: 4 agents — F1 `oracle`, F2 `unspecified-high`, F3 `unspecified-high`, F4 `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.

---

- [x] 1. Repo restructuring + clean package layout

  **What to do**:
  - Delete obsolete scaffold files from the previous narrow-scope plan
  - Create the new repo layout:
    ```
    src/
    ├── __init__.py
    ├── models.py              # Pydantic typed contracts
    ├── protocol.py            # Normalized hybrid ATC protocol
    ├── physics.py             # Deterministic aircraft physics
    ├── airport_schema.py      # Airport topology schema + loader
    ├── state_machine.py       # Full-lifecycle state machine
    ├── api.py                 # REST/API endpoint surface
    ├── rewards.py             # Reward engine + grader primitives
    ├── tasks/                 # Task implementations
    │   ├── __init__.py
    │   ├── registry.py        # Task registry + scenario fixtures
    │   ├── arrival.py         # Landing + handoff + taxi-in + docking
    │   ├── departure.py       # Pushback + taxi-out + release
    │   └── integrated.py      # End-to-end full lifecycle
    ├── phraseology.py         # Renderer/judge for normalized subset
    ├── server/
    │   ├── __init__.py
    │   ├── app.py             # OpenEnv server (reset/step/state)
    │   ├── requirements.txt
    │   └── Dockerfile
    └── visualizer/
        ├── __init__.py
        └── viewer.py          # Read-only 2D synchronized viewer
    tests/
    ├── __init__.py
    └── test_*.py              # TDD tests per module
    openenv.yaml
    inference.py
    pyproject.toml             # uv-managed project config
    ```
  - Initialize `uv` project: `uv init` or create `pyproject.toml` with uv-compatible config
  - Add dependencies via `uv add`: pydantic, fastapi, uvicorn, openenv-core, openai
  - Write RED tests asserting the expected package structure exists

  **Must NOT do**:
  - Do not implement any simulation logic yet
  - Do not keep any old `gatwick_ground_control/` package files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: All subsequent tasks
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] RED test fails for missing expected package structure
  - [ ] GREEN after layout is created
  - [ ] `uv sync` succeeds
  - [ ] `uv run python -c "import src"` succeeds

  **QA Scenarios**:
  ```text
  Scenario: Package structure is correct
    Tool: Bash
    Steps:
      1. Run `uv run python -m pytest tests/test_repo_structure.py -v`
      2. Capture output to `.sisyphus/evidence/task-1-structure.txt`
    Expected Result: All expected files and directories exist
    Evidence: .sisyphus/evidence/task-1-structure.txt

  Scenario: uv sync succeeds
    Tool: Bash
    Steps:
      1. Run `uv sync`
      2. Capture output to `.sisyphus/evidence/task-1-uv-sync.txt`
    Expected Result: Dependencies install without error
    Evidence: .sisyphus/evidence/task-1-uv-sync.txt
  ```

  **Commit**: YES
  - Message: `chore(repo): restructure for clean submission layout`

- [x] 2. Strict typed Pydantic models (extended for full lifecycle)

  **What to do**:
  - Define Pydantic models for:
    - `Action`: structured controller commands (clearance, pushback, taxi route, hold, release)
    - `Observation`: environment response (result, score, phraseology evaluation, issue codes)
    - `State`: full environment state (phase, task_ids, current_task, completed_tasks, episode_id, step_count, metadata)
    - `AircraftState`: deterministic aircraft state (callsign, position, heading, altitude, speed, phase, fuel, assigned_runway, assigned_gate)
    - `LifecyclePhase`: enum covering all phases (approach, landing, arrival_handoff, taxi_in, docking, at_gate, pushback, taxi_out, departure_queue, takeoff, departed)
  - All models use `extra="forbid"` for strict validation
  - Write RED tests for invalid payloads, enum violations, and round-trip serialization

  **Must NOT do**:
  - Do not use `dict[str, Any]` as primary contract
  - Do not allow phraseology text to bypass structured action validation

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7-T16
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] RED tests fail for malformed payloads
  - [ ] GREEN tests pass for valid round-trip serialization
  - [ ] Invalid enum/type combinations are rejected

  **QA Scenarios**:
  ```text
  Scenario: Action round-trips cleanly
    Tool: Bash
    Steps:
      1. Run `uv run python -m pytest tests/test_models.py -v`
      2. Capture output to `.sisyphus/evidence/task-2-roundtrip.txt`
    Expected Result: Valid actions serialize and deserialize without loss
    Evidence: .sisyphus/evidence/task-2-roundtrip.txt

  Scenario: Invalid payload is rejected
    Tool: Bash
    Steps:
      1. Run model validation tests with bad data
      2. Capture output proving ValidationError is raised
    Expected Result: Invalid payloads raise validation failures
    Evidence: .sisyphus/evidence/task-2-invalid.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add full-lifecycle typed contracts`

- [x] 3. Normalized hybrid ATC protocol definitions

  **What to do**:
  - Define a normalized hybrid ATC protocol layer that is airport-agnostic:
    - Standardized clearance types: pushback, taxi, hold_short, cross_runway, line_up, takeoff, landing
    - Standardized readback requirements per clearance type
    - Standardized phraseology templates (not FAA/CAA-locked, but inspired by real procedures)
    - Protocol validation rules (what makes a valid vs invalid instruction)
    - Handoff protocol between ground and tower frequencies
  - Implement protocol validator that checks instruction validity
  - Write RED tests for valid/invalid protocol messages

  **Must NOT do**:
  - Do not lock to FAA or ICAO/UK CAA specifically
  - Do not implement full phraseology corpus — just the v1 subset

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7, T12-T15
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] RED tests fail for invalid protocol messages
  - [ ] GREEN tests pass for valid protocol validation
  - [ ] Protocol is airport-agnostic (no hardcoded airport references)

  **QA Scenarios**:
  ```text
  Scenario: Valid clearance passes protocol validation
    Tool: Bash
    Steps:
      1. Run `uv run python -m pytest tests/test_protocol.py -v`
      2. Capture output to `.sisyphus/evidence/task-3-valid.txt`
    Expected Result: Valid clearances pass validation
    Evidence: .sisyphus/evidence/task-3-valid.txt

  Scenario: Invalid clearance fails protocol validation
    Tool: Bash
    Steps:
      1. Run tests with malformed clearances
      2. Capture output proving rejection
    Expected Result: Invalid clearances are rejected with clear error codes
    Evidence: .sisyphus/evidence/task-3-invalid.txt
  ```

  **Commit**: YES
  - Message: `feat(protocol): add normalized hybrid ATC protocol`

- [x] 4. Deterministic aircraft physics model

  **What to do**:
  - Implement deterministic aircraft physics sufficient for realistic-enough simulation:
    - Glide path model: standard 3-degree glide slope for approach
    - Descent rate model: realistic feet-per-minute descent profiles
    - Surface movement: taxi speed, acceleration/deceleration on ground
    - Heading/altitude tracking: position updates per timestep
    - Wake category separation: small/large/heavy spacing rules
  - All physics are deterministic under fixed seed
  - No flap settings, engine minutiae, or high-fidelity flight dynamics
  - Write RED tests for physics correctness and determinism

  **Must NOT do**:
  - Do not implement full flight dynamics simulation
  - Do not add flap/gear/engine configuration state

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7, T13-T16, T19
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] RED tests fail for unrealistic physics values
  - [ ] GREEN tests pass for correct glide path, descent rate, and surface movement
  - [ ] Deterministic under fixed seed

  **QA Scenarios**:
  ```text
  Scenario: Glide path follows 3-degree slope
    Tool: Bash
    Steps:
      1. Run `uv run python -m pytest tests/test_physics.py -v`
      2. Capture output to `.sisyphus/evidence/task-4-glidepath.txt`
    Expected Result: Aircraft descent follows realistic 3-degree glide path
    Evidence: .sisyphus/evidence/task-4-glidepath.txt

  Scenario: Physics are deterministic under fixed seed
    Tool: Bash
    Steps:
      1. Run same scenario twice with same seed
      2. Compare outputs
    Expected Result: Identical results under same seed
    Evidence: .sisyphus/evidence/task-4-determinism.txt
  ```

  **Commit**: YES
  - Message: `feat(physics): add deterministic aircraft physics`

- [x] 5. Airport schema layer (Gatwick-first, extensible)

  **What to do**:
  - Reuse and adapt the existing airport schema from the previous plan:
    - `AirportSchema` with metadata, nodes, edges, annotations
    - NodeType enum: stand, pushback_spot, taxi_point, hold_short, runway_entry, departure_queue, approach_fix, glide_path
    - EdgeMovement enum: pushback, taxi, queue_join, runway_transition, approach, landing
    - Static JSON fixtures for Gatwick and one dummy airport
    - Loader/validator with required node/edge class enforcement
  - Extend schema to support arrival-side nodes (approach_fix, glide_path, landing_threshold)
  - Write RED tests for schema loading, invalid topology, and extensibility

  **Must NOT do**:
  - Do not hard-code Gatwick into core simulation logic
  - Do not fetch airport data at runtime

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T7, T10, T13-T16, T20
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] RED tests fail for invalid airport schema
  - [ ] GREEN tests pass for Gatwick and dummy airport loading
  - [ ] Schema supports both arrival and departure topology

  **QA Scenarios**:
  ```text
  Scenario: Gatwick schema loads with full lifecycle topology
    Tool: Bash
    Steps:
      1. Run `uv run python -m pytest tests/test_airport_schema.py -v`
      2. Capture output to `.sisyphus/evidence/task-5-gatwick.txt`
    Expected Result: Gatwick schema loads with arrival + departure nodes
    Evidence: .sisyphus/evidence/task-5-gatwick.txt

  Scenario: Extensibility path works for another airport
    Tool: Bash
    Steps:
      1. Run schema tests for dummy airport fixture
      2. Capture output proving same code path works
    Expected Result: Loader is airport-generic
    Evidence: .sisyphus/evidence/task-5-extensibility.txt
  ```

  **Commit**: YES
  - Message: `feat(schema): add full-lifecycle airport schema`

- [x] 6. TDD/test harness + validator smoke scaffolding

  **What to do**:
  - Set up `uv`-managed test configuration with `pytest`
  - Create shared fixture modules for models, airport schema, tasks, and endpoint tests
  - Add helper utilities for stdout capture, local server bootstrapping, and seeded env setup
  - Add local pre-submit validation wrapper aligned with competition validator expectations
  - Write RED tests proving the harness works

  **Must NOT do**:
  - Do not couple tests so tightly to one module that shared harness logic becomes unusable

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1
  - **Blocks**: T8-T20
  - **Blocked By**: None

  **Acceptance Criteria**:
  - [ ] RED tests exist before major modules land
  - [ ] Shared fixtures support all test categories
  - [ ] Single `uv run pytest` command runs the full suite

  **QA Scenarios**:
  ```text
  Scenario: Test harness runs cleanly
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/ -v`
      2. Capture output to `.sisyphus/evidence/task-6-harness.txt`
    Expected Result: Test discovery and execution work without bespoke bootstrapping
    Evidence: .sisyphus/evidence/task-6-harness.txt
  ```

  **Commit**: YES
  - Message: `test(harness): add uv-managed tdd scaffolding`

- [x] 7. Full-lifecycle state machine (landing → departure)

  **What to do**:
  - Implement the core state machine that drives the full operational lifecycle:
    - Phases: approach → landing → arrival_handoff → taxi_in → docking → at_gate → pushback → taxi_out → departure_queue → takeoff → departed
    - State transitions are deterministic and driven by agent actions
    - Each phase has entry conditions, exit conditions, and allowed actions
    - State machine consumes airport schema topology for valid transitions
    - Integrates deterministic aircraft physics for position/heading/altitude updates
  - Write RED tests for every state transition, illegal transitions, and phase boundary conditions

  **Must NOT do**:
  - Do not implement task-specific grading logic here
  - Do not add arrival/TRACON control beyond the landing threshold

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T8-T11, T13-T15, T19
  - **Blocked By**: T2, T3, T4, T5

  **Acceptance Criteria**:
  - [ ] RED tests fail for illegal state transitions
  - [ ] GREEN tests pass for full lifecycle traversal
  - [ ] State machine is deterministic under fixed seed

  **QA Scenarios**:
  ```text
  Scenario: Full lifecycle traversal succeeds
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_state_machine.py::test_full_lifecycle -v`
      2. Capture output to `.sisyphus/evidence/task-7-lifecycle.txt`
    Expected Result: Aircraft progresses through all phases without error
    Evidence: .sisyphus/evidence/task-7-lifecycle.txt

  Scenario: Illegal transition is rejected
    Tool: Bash
    Steps:
      1. Run tests attempting phase skip (e.g., approach → takeoff)
      2. Capture output proving rejection
    Expected Result: Illegal transitions raise errors or are blocked
    Evidence: .sisyphus/evidence/task-7-illegal.txt
  ```

  **Commit**: YES
  - Message: `feat(engine): add full-lifecycle state machine`

- [x] 8. REST/API endpoint surface for agent interaction

  **What to do**:
  - Implement the REST/API layer that exposes the environment to the LLM/RL agent:
    - `POST /reset` — reset environment to initial state, return observation
    - `POST /step` — execute agent action, return observation + reward + done
    - `GET /state` — return current environment state
    - `GET /health` — health check endpoint
    - All endpoints accept/return JSON with strict typed payloads
    - Endpoints are deterministic under fixed seed
  - Write RED tests for endpoint availability, malformed payloads, and correct response shapes

  **Must NOT do**:
  - Do not expose ad hoc endpoint contracts outside the typed model layer
  - Do not hide state in opaque globals that break reproducibility

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T11, T16-T18
  - **Blocked By**: T2, T6, T7

  **Acceptance Criteria**:
  - [ ] RED tests fail for invalid endpoint behavior
  - [ ] GREEN tests pass for reset/step/state flows
  - [ ] Typed payloads propagate correctly through all endpoints

  **QA Scenarios**:
  ```text
  Scenario: Reset/step/state endpoints work end-to-end
    Tool: Bash
    Steps:
      1. Start local server with `uv run python -m src.server.app`
      2. POST to /reset, then POST to /step, then GET /state
      3. Capture responses to `.sisyphus/evidence/task-8-endpoints.txt`
    Expected Result: All endpoints respond with correct typed payloads
    Evidence: .sisyphus/evidence/task-8-endpoints.txt

  Scenario: Malformed payload is rejected cleanly
    Tool: Bash
    Steps:
      1. POST malformed JSON to /step
      2. Capture error response
    Expected Result: Invalid input fails predictably with clear error
    Evidence: .sisyphus/evidence/task-8-malformed.txt
  ```

  **Commit**: YES
  - Message: `feat(api): add REST endpoint surface`

- [x] 9. Reward engine + grader primitives

  **What to do**:
  - Implement the reward/scoring engine that evaluates agent decisions:
    - Safety gate: hard penalties for collisions, runway incursions, conflicting clearances
    - Legality: penalties for protocol violations, invalid clearances, wrong phraseology
    - Completion: bonuses for successful phase transitions (safe landing, safe docking, safe departure)
    - Efficiency: penalties for unnecessary delays, blocked taxiways, excessive queue time
    - Communication: scoring for correct phraseology and readback completeness
    - All scores normalized to `[0.0, 1.0]`
    - Safety failures dominate efficiency gains (unsafe = low score regardless of throughput)
  - Write RED tests for score range enforcement, safety-over-throughput behavior, and normalization

  **Must NOT do**:
  - Do not reward chatter or text verbosity
  - Do not allow efficiency to override safety failures

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T10, T13-T15, T20
  - **Blocked By**: T2, T6, T7

  **Acceptance Criteria**:
  - [ ] RED tests fail when scores exceed `[0.0, 1.0]`
  - [ ] GREEN tests prove unsafe states gate score downward
  - [ ] Shared primitives are reusable by all task graders

  **QA Scenarios**:
  ```text
  Scenario: Score normalization clamps correctly
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_rewards.py -v`
      2. Capture output to `.sisyphus/evidence/task-9-range.txt`
    Expected Result: All scores remain within `[0.0, 1.0]`
    Evidence: .sisyphus/evidence/task-9-range.txt

  Scenario: Safety gate beats throughput reward
    Tool: Bash
    Steps:
      1. Run tests for unsafe-but-efficient vs safe-but-slower scenarios
      2. Capture output showing safety-gated scoring
    Expected Result: Unsafe scenario cannot outscore safe one through efficiency alone
    Evidence: .sisyphus/evidence/task-9-safety.txt
  ```

  **Commit**: YES
  - Message: `feat(rewards): add reward engine and grader primitives`

- [x] 10. Task registry + deterministic scenario fixtures

  **What to do**:
  - Define the task registry that maps task IDs to their implementations:
    - `arrival`: landing + arrival handoff + taxi-in + docking
    - `departure`: pushback + taxi-out + release sequencing
    - `integrated`: end-to-end full lifecycle episode
  - Create deterministic scenario fixtures keyed by task ID and seed
  - Implement seeded scenario builder that reproduces identical initial states under fixed seeds
  - Write RED tests for unknown task IDs, seed determinism, and scenario/task mismatch errors

  **Must NOT do**:
  - Do not blend all task logic into one opaque mega-scenario
  - Do not leave scenario generation nondeterministic under fixed seeds

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T13-T15, T19, T20
  - **Blocked By**: T2, T6, T7, T9

  **Acceptance Criteria**:
  - [ ] RED tests fail for unknown task IDs and inconsistent seeded output
  - [ ] GREEN tests pass for stable task enumeration and repeatable episode construction
  - [ ] All three task IDs are available to inference and graders

  **QA Scenarios**:
  ```text
  Scenario: Task registry lists exactly three tasks
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_tasks.py::test_task_registry -v`
      2. Capture output to `.sisyphus/evidence/task-10-registry.txt`
    Expected Result: Exactly three tasks enumerated: arrival, departure, integrated
    Evidence: .sisyphus/evidence/task-10-registry.txt

  Scenario: Fixed seed yields identical scenario state
    Tool: Bash
    Steps:
      1. Run deterministic seed tests twice with same seed
      2. Compare outputs
    Expected Result: Scenario generation is reproducible under fixed seed
    Evidence: .sisyphus/evidence/task-10-determinism.txt
  ```

  **Commit**: YES
  - Message: `feat(tasks): add task registry and scenario fixtures`

- [x] 11. OpenEnv server/client integration (reset/step/state)

  **What to do**:
  - Wire the environment into OpenEnv-compatible server/client flow:
    - Implement `Environment` base class with `reset()`, `step()`, `state()` methods
    - Ensure typed observations and rewards propagate correctly through the client
    - Default concurrent-session behavior conservatively (explicit session handling)
    - Create FastAPI app that wraps the environment with OpenEnv-style endpoints
    - Write RED tests for endpoint availability, malformed step payloads, and task routing

  **Must NOT do**:
  - Do not expose ad hoc endpoint contracts outside the typed model layer
  - Do not hide grader/task state in opaque globals

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T16-T18, T20
  - **Blocked By**: T2, T6, T7, T8

  **Acceptance Criteria**:
  - [ ] RED tests fail for invalid endpoint behavior
  - [ ] GREEN tests pass for task-specific reset/step/state flows
  - [ ] Typed observations and rewards propagate correctly through the client

  **QA Scenarios**:
  ```text
  Scenario: OpenEnv reset/step/state flow works end-to-end
    Tool: Bash
    Steps:
      1. Start local environment server
      2. Issue reset for a known task/seed, then step, then state
      3. Capture responses to `.sisyphus/evidence/task-11-openenv.txt`
    Expected Result: Endpoints respond successfully with typed payloads
    Evidence: .sisyphus/evidence/task-11-openenv.txt

  Scenario: Invalid step payload is rejected cleanly
    Tool: Bash
    Steps:
      1. Send malformed step payload
      2. Capture rejection behavior
    Expected Result: Invalid input fails predictably without corrupting env state
    Evidence: .sisyphus/evidence/task-11-invalid-step.txt
  ```

  **Commit**: YES
  - Message: `feat(openenv): wire env endpoints and client`

- [x] 12. Phraseology renderer/judge (normalized subset)

  **What to do**:
  - Implement the normalized internal phraseology layer:
    - Renderer: converts structured actions to standardized phraseology strings
    - Judge: scores candidate phraseology against structured action truth
    - Templates for each clearance type: pushback, taxi, hold_short, cross_runway, line_up, takeoff, landing
    - Readback completeness checking
    - Phraseology scoring is separate from operational truth (structured action is ground truth)
  - Write RED tests for correct rendering, mismatch penalties, and missing required elements

  **Must NOT do**:
  - Do not make phraseology text the direct simulator control channel
  - Do not drift into full FAA/ICAO/UK free-form phraseology coverage

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: T13-T15, T17
  - **Blocked By**: T2, T3, T6

  **Acceptance Criteria**:
  - [ ] RED tests fail for malformed or incomplete phraseology outputs
  - [ ] GREEN tests pass for correct rendering/judging from structured actions
  - [ ] Phraseology score is available separately from operational score

  **QA Scenarios**:
  ```text
  Scenario: Structured action renders expected phraseology
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_phraseology.py -v`
      2. Capture output to `.sisyphus/evidence/task-12-renderer.txt`
    Expected Result: Renderer output matches normalized subset rules
    Evidence: .sisyphus/evidence/task-12-renderer.txt

  Scenario: Mismatched phraseology is penalized
    Tool: Bash
    Steps:
      1. Run tests for phraseology that omits required instruction elements
      2. Capture judge output showing lower score
    Expected Result: Incorrect phrasing is caught without corrupting structured action truth
    Evidence: .sisyphus/evidence/task-12-mismatch.txt
  ```

  **Commit**: YES
  - Message: `feat(comms): add normalized phraseology judge`

- [x] 13. Arrival task (landing + handoff + taxi-in + docking)

  **What to do**:
  - Implement the arrival task covering the inbound lifecycle:
    - Landing: aircraft on glide path, descent rate tracking, runway threshold crossing
    - Arrival handoff: transition from tower frequency to ground frequency
    - Taxi-in: route from runway exit to assigned gate/stand
    - Docking: gate arrival, engine shutdown, ground equipment connection
    - Task grader using shared reward primitives
  - Write RED tests for safe arrival success and unsafe arrival failure

  **Must NOT do**:
  - Do not implement en-route/approach control beyond the landing threshold
  - Do not add gate assignment logic (gate is pre-assigned in scenario fixture)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T15, T20
  - **Blocked By**: T2-T7, T9-T12

  **Acceptance Criteria**:
  - [ ] RED tests fail for unsafe landing/handoff/taxi-in/docking
  - [ ] GREEN tests pass for safe arrival with normalized score in `[0.0, 1.0]`
  - [ ] Task exposes deterministic reset/step behavior under fixed seed

  **QA Scenarios**:
  ```text
  Scenario: Safe arrival completes all phases
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_arrival_task.py -v`
      2. Capture output to `.sisyphus/evidence/task-13-safe.txt`
    Expected Result: Aircraft lands, hands off, taxis in, docks safely with score in [0.0, 1.0]
    Evidence: .sisyphus/evidence/task-13-safe.txt

  Scenario: Unsafe arrival is penalized
    Tool: Bash
    Steps:
      1. Run tests for runway incursion or missed handoff
      2. Capture output showing heavy penalty
    Expected Result: Unsafe arrival receives low score regardless of efficiency
    Evidence: .sisyphus/evidence/task-13-unsafe.txt
  ```

  **Commit**: YES
  - Message: `feat(task): add arrival lifecycle benchmark`

- [x] 14. Departure task (pushback + taxi-out + release sequencing)

  **What to do**:
  - Implement the departure task covering the outbound lifecycle:
    - Pushback: approve/defer pushback from gate, apron safety checks
    - Taxi-out: route from gate to departure runway via taxiway network
    - Release sequencing: queue management, runway release timing, handoff to tower
    - Task grader using shared reward primitives
  - Write RED tests for safe departure success and unsafe departure failure

  **Must NOT do**:
  - Do not model airborne takeoff roll beyond runway release
  - Do not reward queue progress if runway access is unsafe

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T15, T17, T20
  - **Blocked By**: T2-T7, T9-T12

  **Acceptance Criteria**:
  - [ ] RED tests fail for unsafe pushback/taxi-out/release
  - [ ] GREEN tests pass for safe departure with normalized score in `[0.0, 1.0]`
  - [ ] Task exposes deterministic reset/step behavior under fixed seed

  **QA Scenarios**:
  ```text
  Scenario: Safe departure completes all phases
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_departure_task.py -v`
      2. Capture output to `.sisyphus/evidence/task-14-safe.txt`
    Expected Result: Aircraft pushes back, taxis out, releases safely with score in [0.0, 1.0]
    Evidence: .sisyphus/evidence/task-14-safe.txt

  Scenario: Unsafe departure is blocked
    Tool: Bash
    Steps:
      1. Run tests for apron conflict or unsafe runway release
      2. Capture output showing rejection or penalty
    Expected Result: Unsafe departure cannot succeed
    Evidence: .sisyphus/evidence/task-14-unsafe.txt
  ```

  **Commit**: YES
  - Message: `feat(task): add departure lifecycle benchmark`

- [x] 15. Integrated full-lifecycle task (end-to-end episode)

  **What to do**:
  - Implement the integrated task combining arrival and departure in one episode:
    - Aircraft arrives, lands, taxis in, docks
    - After turnaround (simulated), pushes back, taxis out, departs
    - Full lifecycle scoring using shared reward primitives
    - Tests agent ability to manage the complete flow, not just isolated phases
  - Write RED tests for full lifecycle success and partial failure scenarios

  **Must NOT do**:
  - Do not add turnaround logistics (refueling, boarding) — simulated as time delay
  - Do not make this task so complex that it becomes a mega-scenario

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T17, T20
  - **Blocked By**: T2-T14

  **Acceptance Criteria**:
  - [ ] RED tests fail for lifecycle failures in any phase
  - [ ] GREEN tests pass for complete end-to-end episode with normalized score
  - [ ] Task exposes deterministic reset/step behavior under fixed seed

  **QA Scenarios**:
  ```text
  Scenario: Full lifecycle completes end-to-end
    Tool: Bash
    Steps:
      1. Run `uv run pytest tests/test_integrated_task.py -v`
      2. Capture output to `.sisyphus/evidence/task-15-full.txt`
    Expected Result: Aircraft completes landing through departure with score in [0.0, 1.0]
    Evidence: .sisyphus/evidence/task-15-full.txt

  Scenario: Phase failure propagates correctly
    Tool: Bash
    Steps:
      1. Run tests where one phase fails (e.g., unsafe taxi-in)
      2. Capture output showing score impact across remaining phases
    Expected Result: Phase failure affects overall score appropriately
    Evidence: .sisyphus/evidence/task-15-partial.txt
  ```

  **Commit**: YES
  - Message: `feat(task): add integrated full-lifecycle benchmark`

- [x] 16. Read-only 2D visualizer synchronized to env state

  **What to do**:
  - Implement a read-only 2D visualizer that mirrors environment state:
    - Airport topology rendering: runways, taxiways, gates, hold-short points
    - Aircraft rendering: simple shapes with callsign, heading, altitude, speed labels
    - State synchronization: reads environment state via API or direct state access
    - Simple animation: smooth position interpolation between state updates
    - Debug overlays: occupancy zones, conflict areas, decision states
    - No interactive controls — purely a viewer
  - Write RED tests for state synchronization and rendering correctness

  **Must NOT do**:
  - Do not add interactive controls or operator input
  - Do not couple visualizer logic to environment simulation logic

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T17
  - **Blocked By**: T2, T4, T5, T7, T8, T11

  **Acceptance Criteria**:
  - [ ] RED tests fail for state synchronization mismatches
  - [ ] GREEN tests pass for correct rendering of environment state
  - [ ] Visualizer updates smoothly without blocking environment execution

  **QA Scenarios**:
  ```text
  Scenario: Visualizer synchronizes to environment state
    Tool: Bash
    Steps:
      1. Run environment with visualizer attached
      2. Capture screenshot to `.sisyphus/evidence/task-16-sync.png`
      3. Verify aircraft positions match environment state
    Expected Result: Visualizer accurately reflects environment state
    Evidence: .sisyphus/evidence/task-16-sync.png

  Scenario: Visualizer is read-only
    Tool: Bash
    Steps:
      1. Attempt to interact with visualizer controls
      2. Verify no state changes occur in environment
    Expected Result: Visualizer has no control capability
    Evidence: .sisyphus/evidence/task-16-readonly.txt
  ```

  **Commit**: YES
  - Message: `feat(viz): add read-only 2D visualizer`

- [x] 17. Root inference.py + competition logging

  **What to do**:
  - Implement the root-level `inference.py` matching the exact sample format:
    - Uses `asyncio` + `OpenAI` client with `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
    - Loads env via `from openenv.core.client import EnvClient` pattern (or custom client matching `from_docker_image`)
    - Emits **exact** stdout format per sample:
      - `[START] task=<task_name> env=<benchmark> model=<model_name>`
      - `[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>`
      - `[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>`
    - Runs all three tasks (arrival=hard, departure=easy, integrated=medium) sequentially
    - Handles errors gracefully, always emitting `[END]` line
    - Score normalization: `score = sum(rewards) / MAX_TOTAL_REWARD`, clamped to `[0.0, 1.0]`
  - Write RED tests for formatting correctness, env-var handling, and failure-path `[END]` emission

  **Must NOT do**:
  - Do not deviate from exact field names/order/format (e.g., `score` must be 3 decimal places, `reward` must be 2)
  - Do not bypass the OpenAI client requirement
  - Do not use synchronous `env.reset()` — must use `await env.reset()`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3
  - **Blocks**: T18, T20
  - **Blocked By**: T1, T6, T11-T16

  **Acceptance Criteria**:
  - [ ] RED tests fail on any log formatting deviation
  - [ ] GREEN tests pass for env-var handling, step logging, and guaranteed end-line emission
  - [ ] `uv run python inference.py` completes without error against local environment
  - [ ] Output matches sample format exactly (field names, decimal places, boolean casing)

  **QA Scenarios**:
  ```text
  Scenario: Inference emits exact required log sequence
    Tool: Bash
    Steps:
      1. Run `uv run python inference.py`
      2. Capture stdout to `.sisyphus/evidence/task-17-stdout.txt`
      3. Assert lines match:
         - `^\[START\] task=.+ env=.+ model=.+$`
         - `^\[STEP\] step=\d+ action=.+ reward=\d+\.\d{2} done=(true|false) error=.+$`
         - `^\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=.+$`
    Expected Result: Output format matches competition requirements exactly
    Evidence: .sisyphus/evidence/task-17-stdout.txt

  Scenario: Failure path still emits [END]
    Tool: Bash
    Steps:
      1. Run targeted inference failure-path test
      2. Capture output proving [END] is still emitted after exception
    Expected Result: Script always emits terminal output
    Evidence: .sisyphus/evidence/task-17-failure.txt
  ```

  **Commit**: YES
  - Message: `feat(submit): add competition inference script`

- [x] 18. Demo script + HF Space landing page

  **What to do**:
  - Create `demo.py` — a short, non-interactive script that:
    - Runs a brief simulation episode against the environment
    - Outputs a summary of the episode (tasks completed, scores, key decisions)
    - Can be run standalone without an LLM (uses deterministic baseline agent)
    - Produces output suitable for HF Space landing page display
  - Create `requirements.txt` at repo root listing all dependencies
  - Write tests proving demo runs without error

  **Must NOT do**:
  - Do not require LLM inference for the demo to work
  - Do not make the demo interactive

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T20
  - **Blocked By**: T13, T14, T15

  **Acceptance Criteria**:
  - [ ] `uv run python demo.py` completes without error
  - [ ] `requirements.txt` exists at repo root with all dependencies
  - [ ] Demo output is human-readable and shows environment functionality

  **QA Scenarios**:
  ```text
  Scenario: Demo script runs successfully
    Tool: Bash
    Steps:
      1. Run `uv run python demo.py`
      2. Capture output to `.sisyphus/evidence/task-18-demo.txt`
    Expected Result: Demo completes and shows episode summary
    Evidence: .sisyphus/evidence/task-18-demo.txt
  ```

  **Commit**: YES
  - Message: `feat(demo): add demo script and requirements.txt`

- [x] 19. Docker/HF Space packaging + health/reset validation

  **What to do**:
  - Build the `Dockerfile` at repo root (matching `prevalidation.sh` expectations):
    - Multi-stage Docker build with uv for dependency installation
    - Exposes port 8000
    - Health check endpoint returns 200 with JSON payload
    - `POST /reset` endpoint responds correctly for HF Space ping
    - Environment variables (`API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`) are configurable
  - Write RED tests or smoke checks for build failure, startup failure, and reset endpoint failure
  - Ensure Dockerfile is discoverable by `prevalidation.sh` (root or `server/` directory)

  **Must NOT do**:
  - Do not rely on manual local state or unpinned assumptions that break in containerized execution
  - Do not leave `/reset` behavior untested until submission day

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T20
  - **Blocked By**: T1, T6, T8, T11, T17

  **Acceptance Criteria**:
  - [ ] Docker build succeeds locally
  - [ ] Local container responds successfully to health and reset checks
  - [ ] Packaging path aligns with `prevalidation.sh` expectations

  **QA Scenarios**:
  ```text
  Scenario: Docker image builds successfully
    Tool: Bash
    Steps:
      1. Run `docker build -t atc-ground-control .`
      2. Capture build output to `.sisyphus/evidence/task-19-build.txt`
    Expected Result: Build completes successfully
    Evidence: .sisyphus/evidence/task-19-build.txt

  Scenario: Container responds to reset
    Tool: Bash
    Steps:
      1. Run container and POST to `/reset`
      2. Capture response body/status to `.sisyphus/evidence/task-19-reset.txt`
      3. Assert HTTP 200
    Expected Result: Reset returns 200 and valid payload
    Evidence: .sisyphus/evidence/task-19-reset.txt
  ```

  **Commit**: YES
  - Message: `chore(deploy): add docker and space packaging`

- [x] 20. Runtime/resource hardening + determinism

  **What to do**:
  - Measure and harden the environment and inference path to fit 2 vCPU / 8 GB and under-20-minute inference constraints:
    - Reduce nondeterminism in scenario generation, stepping, grading, and inference orchestration
    - Optimize physics calculations, state machine transitions, and reward computation
    - Eliminate memory leaks and unnecessary computation hotspots
  - Write RED tests or smoke checks for timeouts, unstable seeds, and runaway memory/time behavior

  **Must NOT do**:
  - Do not overfit to one machine by using hidden local caches or oversized defaults
  - Do not leave performance tuning until after the benchmark assembly step

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: T21
  - **Blocked By**: T4, T7, T10

  **Acceptance Criteria**:
  - [ ] Repeated seeded runs produce stable outcomes
  - [ ] Inference and core smoke tests complete within the submission budget
  - [ ] No obvious unnecessary memory/runtime hotspots remain in the v1 path

  **QA Scenarios**:
  ```text
  Scenario: Seeded benchmark run is stable across repeats
    Tool: Bash
    Steps:
      1. Execute same seeded smoke episode twice
      2. Capture outputs to paired evidence files
      3. Compare key results including task ID, score range, and terminal state
    Expected Result: Repeat runs are materially identical for seeded scenarios
    Evidence: .sisyphus/evidence/task-20-seed-repeat.txt

  Scenario: Inference stays within runtime budget
    Tool: Bash
    Steps:
      1. Time `uv run python inference.py` under local smoke conditions
      2. Capture the timing result and stdout
    Expected Result: Runtime remains under the target budget
    Evidence: .sisyphus/evidence/task-20-runtime.txt
  ```

  **Commit**: YES
  - Message: `perf(core): harden runtime and determinism`

- [x] 21. Benchmark assembly + pre-submit validation flow

  **What to do**:
  - Assemble the final benchmark package so the three tasks, graders, inference path, and validation workflow work as one submission artifact:
    - Wire all three tasks (departure=easy, integrated=medium, arrival=hard) into a single benchmark discovery/enumeration path
    - Ensure benchmark discovery/enumeration, score reporting, and local pre-submit validation match competition expectations
    - Implement integration reporting that surfaces per-task scores in `[0.0, 1.0]`
    - Add local pre-submit flow that chains tests, validate, inference, task enumeration, and score checks
    - Run `prevalidation.sh` locally against the repo to confirm all 3 checks pass
  - Write RED integration tests for missing task registration, grader score range regressions, and submission-path regressions

  **Must NOT do**:
  - Do not ship loosely connected pieces that only work in isolation
  - Do not treat the validator as the only integration check

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: T1, T5, T6, T9, T10, T13-T20

  **Acceptance Criteria**:
  - [ ] Integration tests pass for task enumeration and per-task scoring
  - [ ] Every task reports a score in `[0.0, 1.0]`
  - [ ] Local pre-submit run exercises validator-adjacent checks and succeeds
  - [ ] `prevalidation.sh` passes all 3 checks (HF ping, Docker build, openenv validate)

  **QA Scenarios**:
  ```text
  Scenario: All three tasks enumerate and grade
    Tool: Bash
    Steps:
      1. Run the integration suite or task-enumeration smoke command
      2. Capture output listing the three tasks and their score-producing runs
      3. Assert each reported score lies in `[0.0, 1.0]`
    Expected Result: All tasks are discoverable and gradable
    Evidence: .sisyphus/evidence/task-21-enumeration.txt

  Scenario: Prevalidation script passes all checks
    Tool: Bash
    Steps:
      1. Run `bash samplematerial/prevalidation.sh http://localhost:8000 .`
      2. Capture output to `.sisyphus/evidence/task-21-prevalidate.txt`
    Expected Result: All 3/3 checks pass
    Evidence: .sisyphus/evidence/task-21-prevalidate.txt
  ```

  **Commit**: YES
  - Message: `chore(release): assemble benchmark and validation flow`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Verify every deliverable exists, evidence files exist, and all must-have/must-not-have conditions are satisfied.

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run type checks, tests, lints. Inspect for brittle grader logic, weak typing, unused code, and scope creep.

- [x] F3. **Real Manual QA** — `unspecified-high`
  Execute every task-level QA scenario plus integrated inference, Docker, endpoint, and visualizer flows. Save evidence under `.sisyphus/evidence/final-qa/`.

- [x] F4. **Scope Fidelity Check** — `deep`
  Confirm full lifecycle coverage, normalized protocol, read-only visualizer, OpenEnv compliance, and no scope creep into en-route/TRACON or minutiae simulation.

---

## Commit Strategy

- **1**: `chore(repo): restructure for clean submission layout`
- **2**: `feat(core): add full-lifecycle typed contracts`
- **3**: `feat(env): add state machine, physics, and API surface`
- **4**: `feat(tasks): add arrival, departure, and integrated tasks`
- **5**: `feat(viz): add read-only 2D visualizer`
- **6**: `chore(submit): finalize inference, docker, and validation`

---

## Success Criteria

### Verification Commands
```bash
openenv validate
python inference.py
docker build .
```

### Final Checklist
- [ ] Full lifecycle works: landing → arrival handoff → taxi-in → docking → pushback → taxi-out → departure
- [ ] Deterministic aircraft as environment actors
- [ ] LLM/RL agent orchestrates via REST/API
- [ ] Normalized hybrid ATC protocol enforced
- [ ] Read-only 2D visualizer synchronized to environment state
- [ ] OpenEnv compliance: openenv.yaml, typed models, reset/step/state
- [ ] Competition-ready: inference.py, Docker, 3+ graded tasks, [0.0, 1.0] scores
- [ ] TDD tests pass across all modules
- [ ] Runtime <20 minutes on 2 vCPU / 8 GB RAM
