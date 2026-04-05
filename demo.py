#!/usr/bin/env python3
"""Demo script showing the 911 dispatch supervisor environment in action.

This non-interactive demo runs a deterministic episode using OpenEnvEnvironment
directly (no LLM/API server required). It performs a scripted triage sequence
on the multi-incident task.
"""

import asyncio
import sys

from src.models import Action, DispatchAction
from src.openenv_environment import OpenEnvEnvironment


async def run_demo_episode(
    seed: int = 42, task_id: str = "multi_incident", max_steps: int = 120
) -> dict:
    """Run a deterministic demo episode."""
    print("=" * 60)
    print("911 DISPATCH SUPERVISOR - DEMO EPISODE")
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
        print(f"Initial incidents: {len(state.incidents)}")
        print(f"Initial units: {len(state.units)}")
        for inc in sorted(state.incidents.values(), key=lambda i: i.incident_id):
            print(
                f"  - {inc.incident_id}: {inc.incident_type.value} {inc.severity.value} ({inc.status.value})"
            )
        print("-" * 60)

        # Track episode progress
        step_count = 0
        total_reward = 0.0
        rewards = []
        errors = []

        scripted_actions = [
            Action(action_type=DispatchAction.DISPATCH, unit_id="MED-1", incident_id="INC-002"),
            Action(action_type=DispatchAction.DISPATCH, unit_id="ENG-1", incident_id="INC-001"),
            Action(action_type=DispatchAction.DISPATCH, unit_id="LAD-1", incident_id="INC-001"),
            Action(action_type=DispatchAction.DISPATCH, unit_id="PAT-1", incident_id="INC-003"),
        ]

        for action in scripted_actions:
            step_count += 1
            try:
                obs, reward, done = await env.step(action)
                rewards.append(reward)
                total_reward += reward

                print(
                    f"[STEP {step_count}] Action: {action.action_type.value} {action.unit_id}->{action.incident_id} "
                    f"Reward: {reward:.4f} Done: {done} Issues: {obs.issues}"
                )

                if done:
                    break
            except Exception as e:
                errors.append(f"Step {step_count}: {str(e)}")
                print(f"[STEP {step_count}] ERROR: {e}")
                break

        # Continue stepping with any legal actions until done/max_steps.
        while step_count < max_steps:
            legal = env.legal_actions()
            if not legal:
                break
            action = legal[0]
            step_count += 1
            obs, reward, done = await env.step(action)
            rewards.append(reward)
            total_reward += reward
            if done:
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
        print(f"Steps Taken:   {step_count}")
        print(f"Total Reward:  {total_reward:.4f}")
        print(f"Final Score:   {final_score:.4f}")
        print(f"Active incidents: {sum(1 for i in final_state.incidents.values() if i.status.value != 'RESOLVED')}")

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
            "steps": step_count,
            "total_reward": total_reward,
            "final_score": final_score,
            "errors": errors,
        }

    finally:
        env.close()


def main() -> int:
    """Main entry point for demo script."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     911 DISPATCH SUPERVISOR DEMO                            ║")
    print("║     City-wide emergency dispatch RL environment              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\n")

    try:
        result = asyncio.run(
            run_demo_episode(seed=42, task_id="multi_incident", max_steps=120)
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
