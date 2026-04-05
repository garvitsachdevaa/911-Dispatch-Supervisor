"""Runtime and memory budget smoke tests for dispatch environment."""

from __future__ import annotations

import gc
import time
import tracemalloc

from src.openenv_environment import OpenEnvEnvironment


def _run_steps(task_id: str, seed: int, num_steps: int) -> None:
    import asyncio

    env = OpenEnvEnvironment(task_id=task_id, seed=seed)
    asyncio.run(env.reset())
    for _ in range(num_steps):
        legal = env.legal_actions()
        if not legal:
            break
        asyncio.run(env.step(legal[0]))
    env.close()


def test_50_steps_under_30_seconds() -> None:
    gc.collect()
    start = time.perf_counter()
    _run_steps("multi_incident", seed=42, num_steps=50)
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0


def test_no_large_memory_growth_over_50_steps() -> None:
    gc.collect()
    tracemalloc.start()

    _run_steps("single_incident", seed=42, num_steps=1)
    snap1 = tracemalloc.take_snapshot()
    mem1 = sum(s.size for s in snap1.statistics("lineno"))

    _run_steps("single_incident", seed=42, num_steps=50)
    snap2 = tracemalloc.take_snapshot()
    mem2 = sum(s.size for s in snap2.statistics("lineno"))

    tracemalloc.stop()

    growth_mb = (mem2 - mem1) / (1024 * 1024)
    assert growth_mb < 25.0
