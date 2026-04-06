"""Run baseline inference matrix (random + Open LLM) and summarize variance.

Usage examples:
  python scripts/run_baseline_matrix.py --random-runs 1 --llm-runs 0
  python scripts/run_baseline_matrix.py --random-runs 1 --llm-runs 3 --output-json baseline_report.json

Environment variables:
  API_BASE_URL, MODEL_NAME
  OPENAI_API_KEY or HF_TOKEN (required when --llm-runs > 0)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

START_RE = re.compile(r"^\[START\]\s+task=(\S+)\s+env=(\S+)\s+model=(\S+)$")
END_RE = re.compile(
    r"^\[END\]\s+success=(true|false)\s+steps=(\d+)\s+score=([0-9]*\.?[0-9]+)\s+rewards=(.*)$"
)


@dataclass
class TaskEpisode:
    task_id: str
    success: bool
    steps: int
    score: float


@dataclass
class RunResult:
    lane: str
    run_index: int
    runtime_seconds: float
    tasks: list[TaskEpisode]
    return_code: int
    stderr: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _required_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _extract_task_episodes(stdout: str) -> list[TaskEpisode]:
    episodes: list[TaskEpisode] = []
    current_task: str | None = None

    for line in stdout.splitlines():
        start_match = START_RE.match(line)
        if start_match:
            current_task = start_match.group(1)
            continue

        end_match = END_RE.match(line)
        if end_match:
            task_id = current_task or f"unknown-{len(episodes) + 1}"
            episodes.append(
                TaskEpisode(
                    task_id=task_id,
                    success=end_match.group(1) == "true",
                    steps=int(end_match.group(2)),
                    score=float(end_match.group(3)),
                )
            )
            current_task = None

    return episodes


def _run_inference(lane: str, run_index: int, timeout_seconds: int) -> RunResult:
    env = os.environ.copy()
    env.setdefault("API_BASE_URL", "https://api.openai.com/v1")
    env.setdefault("MODEL_NAME", "baseline-model")

    if lane == "random":
        env["USE_RANDOM"] = "true"
        env.setdefault("OPENAI_API_KEY", "dummy-token")
    else:
        env["USE_RANDOM"] = "false"
        if not (env.get("OPENAI_API_KEY") or env.get("HF_TOKEN")):
            raise RuntimeError(
                "OPENAI_API_KEY or HF_TOKEN is required for Open LLM runs"
            )

    cmd = [sys.executable, "inference.py"]
    started = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=str(_project_root()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=timeout_seconds,
    )
    runtime = time.monotonic() - started

    tasks = _extract_task_episodes(proc.stdout)

    return RunResult(
        lane=lane,
        run_index=run_index,
        runtime_seconds=runtime,
        tasks=tasks,
        return_code=proc.returncode,
        stderr=proc.stderr.strip(),
    )


def _summarize(runs: list[RunResult]) -> dict[str, dict[str, float]]:
    by_task: dict[str, list[float]] = {}
    for run in runs:
        for ep in run.tasks:
            by_task.setdefault(ep.task_id, []).append(ep.score)

    summary: dict[str, dict[str, float]] = {}
    for task_id, scores in sorted(by_task.items()):
        mean_score = statistics.mean(scores)
        stdev_score = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        summary[task_id] = {
            "runs": float(len(scores)),
            "mean": round(mean_score, 6),
            "std": round(stdev_score, 6),
            "min": round(min(scores), 6),
            "max": round(max(scores), 6),
        }
    return summary


def _print_summary(title: str, runs: list[RunResult]) -> None:
    print(f"\n=== {title} ===")
    if not runs:
        print("No runs executed")
        return

    summary = _summarize(runs)
    for task_id, metrics in summary.items():
        print(
            f"{task_id:16s} runs={int(metrics['runs'])} "
            f"mean={metrics['mean']:.3f} std={metrics['std']:.3f} "
            f"min={metrics['min']:.3f} max={metrics['max']:.3f}"
        )

    total_runtime = sum(r.runtime_seconds for r in runs)
    failures = [r for r in runs if r.return_code != 0]
    print(f"total_runtime_seconds={total_runtime:.2f}")
    print(f"failed_runs={len(failures)}")


def _to_jsonable(runs: list[RunResult]) -> list[dict]:
    serialized: list[dict] = []
    for run in runs:
        entry = asdict(run)
        entry["tasks"] = [asdict(t) for t in run.tasks]
        serialized.append(entry)
    return serialized


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline matrix for inference.py")
    parser.add_argument("--random-runs", type=int, default=1)
    parser.add_argument("--llm-runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    os.environ.setdefault("API_BASE_URL", "https://api.openai.com/v1")
    os.environ.setdefault("MODEL_NAME", "nvidia/Nemotron-3-Super-49B-v1")

    _required_var("API_BASE_URL")
    _required_var("MODEL_NAME")

    random_runs: list[RunResult] = []
    llm_runs: list[RunResult] = []

    try:
        for idx in range(1, args.random_runs + 1):
            print(f"Running random baseline {idx}/{args.random_runs}...")
            random_runs.append(_run_inference("random", idx, args.timeout_seconds))

        for idx in range(1, args.llm_runs + 1):
            print(f"Running Open LLM baseline {idx}/{args.llm_runs}...")
            llm_runs.append(_run_inference("llm", idx, args.timeout_seconds))
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    _print_summary("Random Baseline", random_runs)
    _print_summary("Open LLM Baseline", llm_runs)

    all_runs = random_runs + llm_runs

    if args.output_json:
        report = {
            "api_base_url": os.environ.get("API_BASE_URL", ""),
            "model_name": os.environ.get("MODEL_NAME", ""),
            "random_summary": _summarize(random_runs),
            "llm_summary": _summarize(llm_runs),
            "runs": _to_jsonable(all_runs),
        }
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {out_path}")

    failures = [r for r in all_runs if r.return_code != 0]
    if failures:
        print("\nOne or more runs failed:")
        for run in failures:
            print(f"- lane={run.lane} run={run.run_index} rc={run.return_code}")
            if run.stderr:
                print(run.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
