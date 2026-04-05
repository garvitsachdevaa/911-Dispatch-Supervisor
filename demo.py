#!/usr/bin/env python3
"""Demo script showing the ATC Ground Control environment in action.

This non-interactive demo runs a deterministic episode using the OpenEnvEnvironment
directly, without requiring an LLM or API server. It demonstrates the full
arrival lifecycle: approach → landing → handoff → taxi-in → docking.

Usage:
    uv run python demo.py
"""

import asyncio
import sys

from src.models import Action, ClearanceType
from src.openenv_environment import OpenEnvEnvironment


def get_legal_action_for_phase(env: OpenEnvEnvironment) -> Action | None:
    """Get a legal action for the current environment phase.

    Args:
        env: The OpenEnv environment instance.

    Returns:
        A legal Action for the current phase, or None if no action is available.
    """
    from src.state_machine import FullLifecycleStateMachine

    state = env.state()
    aircraft = list(state.aircraft.values())[0] if state.aircraft else None
    if not aircraft:
        return None

    # Get legal actions from the state machine's internal logic
    # We'll use a simple strategy based on the current phase
    phase = state.phase

    if phase.value == "approach":
        return Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign=aircraft.callsign,
            runway="27L",
        )
    elif phase.value == "landing":
        return Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign=aircraft.callsign,
            runway="27L",
        )
    elif phase.value == "arrival_handoff":
        return Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=aircraft.callsign,
            route=["TAXIWAY_C", "TAXIWAY_D", "GATE_B2"],
        )
    elif phase.value == "taxi_in":
        return Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=aircraft.callsign,
            route=["GATE_B2"],
        )
    elif phase.value == "at_gate":
        return Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=aircraft.callsign,
            pushback_direction="back",
        )
    elif phase.value == "pushback":
        return Action(
            clearance_type=ClearanceType.PUSHBACK,
            target_callsign=aircraft.callsign,
            pushback_direction="back",
        )
    elif phase.value == "taxi_out":
        return Action(
            clearance_type=ClearanceType.TAXI,
            target_callsign=aircraft.callsign,
            route=["DEPARTURE_QUEUE"],
        )
    elif phase.value == "departure_queue":
        return Action(
            clearance_type=ClearanceType.LINE_UP,
            target_callsign=aircraft.callsign,
        )
    elif phase.value == "takeoff":
        return Action(
            clearance_type=ClearanceType.TAKEOFF,
            target_callsign=aircraft.callsign,
        )

    return None


async def run_demo_episode(
    seed: int = 42, task_id: str = "arrival", max_steps: int = 300
) -> dict:
    """Run a deterministic demo episode.

    Args:
        seed: Random seed for deterministic behavior.
        task_id: Task identifier (arrival, departure, integrated).
        max_steps: Maximum number of steps before forcing termination.

    Returns:
        Dictionary containing episode summary information.
    """
    print("=" * 60)
    print("ATC GROUND CONTROL - DEMO EPISODE")
    print("=" * 60)
    print(f"Task: {task_id}")
    print(f"Seed: {seed}")
    print("-" * 60)

    # Initialize environment
    env = OpenEnvEnvironment(task_id=task_id, seed=seed)

    try:
        # Reset environment
        observation = await env.reset()
        state = env.state()

        print(f"Episode ID: {state.episode_id}")
        print(f"Initial Phase: {state.phase.value}")
        print(
            f"Aircraft: {list(state.aircraft.values())[0].callsign if state.aircraft else 'None'}"
        )
        print("-" * 60)

        # Track episode progress
        phases_completed = []
        step_count = 0
        total_reward = 0.0
        rewards = []
        errors = []

        # Run episode
        while step_count < max_steps:
            step_count += 1

            # Get current state
            current_state = env.state()

            # Get legal action for current phase
            action = get_legal_action_for_phase(env)

            if action is None:
                # No action available - check if terminal
                if current_state.phase.value == "departed":
                    print(
                        f"\n[STEP {step_count}] Episode complete (natural termination)"
                    )
                    break
                else:
                    # Try to advance with a default action
                    aircraft = (
                        list(current_state.aircraft.values())[0]
                        if current_state.aircraft
                        else None
                    )
                    if aircraft:
                        action = Action(
                            clearance_type=ClearanceType.TAXI,
                            target_callsign=aircraft.callsign,
                            route=[],
                        )
                    else:
                        errors.append(f"Step {step_count}: No aircraft found")
                        break

            # Execute step
            try:
                obs, reward, done = await env.step(action)
                rewards.append(reward)
                total_reward += reward

                print(f"[STEP {step_count}] Phase: {current_state.phase.value}")
                print(
                    f"         Action: {action.clearance_type.value} -> {action.target_callsign}"
                )
                print(f"         Reward: {reward:.4f} | Done: {done}")

                # Track phase completions
                new_state = env.state()
                if new_state.phase != current_state.phase:
                    phases_completed.append(current_state.phase.value)
                    print(
                        f"         *** PHASE COMPLETE: {current_state.phase.value} ***"
                    )

                if done:
                    print(f"\n[STEP {step_count}] Episode terminated")
                    break

            except Exception as e:
                errors.append(f"Step {step_count}: {str(e)}")
                print(f"[STEP {step_count}] ERROR: {e}")
                break

        # Final state
        final_state = env.state()

        # Calculate final score
        final_score = min(1.0, total_reward)

        print("-" * 60)
        print("EPISODE SUMMARY")
        print("-" * 60)
        print(f"Task ID:       {task_id}")
        print(f"Episode ID:    {final_state.episode_id}")
        print(f"Final Phase:   {final_state.phase.value}")
        print(f"Steps Taken:   {step_count}")
        print(f"Total Reward:  {total_reward:.4f}")
        print(f"Final Score:   {final_score:.4f}")
        print(f"Phases Completed: {len(phases_completed)}")
        for i, phase in enumerate(phases_completed, 1):
            print(f"  {i}. {phase}")

        if errors:
            print(f"\nErrors encountered: {len(errors)}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"\nErrors: None")

        print("=" * 60)

        return {
            "task_id": task_id,
            "episode_id": final_state.episode_id,
            "final_phase": final_state.phase.value,
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": final_score,
            "phases_completed": phases_completed,
            "errors": errors,
        }

    finally:
        env.close()


def main() -> int:
    """Main entry point for demo script."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     GRADIENT ASCENT - ATC GROUND CONTROL DEMO               ║")
    print("║     LLM-powered airport ground control — full lifecycle      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n")

    try:
        result = asyncio.run(
            run_demo_episode(seed=42, task_id="arrival", max_steps=300)
        )

        print("\n[SUCCESS] Demo episode completed successfully!")
        return 0

    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
