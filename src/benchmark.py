"""Benchmark module for running 911 dispatch RL tasks."""

from __future__ import annotations

import asyncio
import random
from typing import Any

from src.models import Action, DispatchAction
from src.openenv_environment import OpenEnvEnvironment
from src.rewards import TaskGrader
from src.tasks.registry import TaskRegistry


def list_tasks() -> list[dict[str, Any]]:
    tasks = TaskRegistry.list_tasks()
    return [
        {"task_id": t.task_id, "name": t.name, "difficulty": t.difficulty}
        for t in tasks
    ]


async def _run_episode_async(task_id: str, seed: int) -> tuple[float, list[float]]:
    env = OpenEnvEnvironment(task_id=task_id, seed=seed)
    rewards: list[float] = []
    final_state = None

    try:
        await env.reset()
        final_state = env.state()

        rng = random.Random(seed)
        for _ in range(1000):
            legal_actions = env.legal_actions()
            if legal_actions:
                action = rng.choice(legal_actions)
            else:
                # Fallback: attempt to dispatch the first unit to the first incident.
                st = env.state()
                if not st.units or not st.incidents:
                    break
                unit_id = next(iter(st.units.keys()))
                incident_id = next(iter(st.incidents.keys()))
                action = Action(
                    action_type=DispatchAction.DISPATCH,
                    unit_id=unit_id,
                    incident_id=incident_id,
                )

            obs, reward, done = await env.step(action)
            rewards.append(reward)

            final_state = env.state()

            if done:
                break
    finally:
        env.close()

    if final_state is None:
        from src.models import State

        final_state = State(
            units={},
            incidents={},
            episode_id="",
            step_count=0,
            task_id=task_id,
            city_time=0.0,
            metadata={},
        )

    # Score episodes the same way as the OpenEnv evaluation path:
    # a normalized aggregate of per-step rewards.
    final_score = TaskGrader().grade_episode(rewards, task_id=task_id)
    return final_score, rewards


def run_task(task_id: str, seed: int) -> dict[str, Any]:
    TaskRegistry.get(task_id)
    final_score, rewards = asyncio.run(_run_episode_async(task_id, seed))
    return {
        "task_id": task_id,
        "seed": seed,
        "score": max(0.0, min(1.0, final_score)),
        "rewards": rewards,
    }


def run_all() -> dict[str, float]:
    scores: dict[str, float] = {}
    for task in TaskRegistry.list_tasks():
        result = run_task(task.task_id, hash(task.task_id) % 10000)
        scores[task.task_id] = result["score"]
    return scores


if __name__ == "__main__":
    print("Available tasks:")
    for task in list_tasks():
        print(f"  - {task['task_id']}: {task['name']} ({task['difficulty']})")
    print("\nRunning all tasks...")
    scores = run_all()
    print("\nScores:")
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.3f}")
