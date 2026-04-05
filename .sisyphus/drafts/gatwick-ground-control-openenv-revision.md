# Draft: Gatwick Ground Control OpenEnv Revision

## Requirements (confirmed)
- The prior benchmark scope is too narrow and must be revised.
- Deterministic aircraft remain environment-side actors; the LLM/RL agent remains the orchestrator.
- The system should operate through a REST/API-oriented environment interface.
- Scope must now include the full operational flow:
  - landing
  - arrival-side handoffs
  - taxi-in
  - docking / gate arrival protocols
  - pushback
  - taxi-out
  - departure sequencing / release
- A demo layer is also required, not just the benchmark core.
- The demo layer should include:
  - simple 2D aircraft rendering
  - callsign / heading / direction / altitude display
  - runway and taxi visualization
  - simple animation
  - synchronization with benchmark/environment state
- Physics should be realistic enough for:
  - accurate glide path
  - realistic rate of descent / heading behavior
  - no need for low-level aircraft configuration details like flap settings

## Technical Decisions
- Current likely architecture: one benchmark core plus one synchronized visualization/demo layer.
- Deterministic environment-side aircraft behavior is still the correct baseline.

## Research Findings
- The previous plan optimized strongly for competition reliability by excluding arrivals and visualization.
- That is no longer sufficient for the revised hackathon scope.

## Open Questions
- None — all scope decisions are now resolved.

## Resolved Decisions
- **Unified RL environment**: The user clarified this is one unified RL training environment, not separate benchmark/demo tracks. The "demo" is simply a read-only visualizer synchronized to environment state.
- **Protocol standard**: Normalized hybrid subset — airport-agnostic, generalized ATC procedures that work across any airport without FAA/CAA lock-in.
- **Visualizer role**: Read-only sync view — mirrors environment state for human understanding, no control capability.
- **Reference repos**: All three (`bluesky/`, `ai-airport-simulation/`, `atc-reinforcement-learning/`) have been reviewed and deleted. All valuable patterns were already extracted into the current codebase.

## Repo Cleanup
- Reference repos removed from workspace root.
- Next step: restructure remaining files into a clean, submission-ready layout.

## Scope Boundaries
- INCLUDE: benchmark core + demo layer, arrival and departure lifecycle, realistic-but-simplified aircraft movement, REST/API-driven architecture.
- EXCLUDE: low-level aircraft systems like flap settings or high-fidelity flight dynamics beyond what is needed for believable glide path/descent and surface motion.

## Session Notes
- User requested: continue immediately if next steps are clear; otherwise pause and ask targeted clarification.
- Planner status: next step is execution handoff against `.sisyphus/plans/atc-ground-control-rl-environment.md` since scope and plan are already established.
