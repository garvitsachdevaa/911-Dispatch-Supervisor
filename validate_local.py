#!/usr/bin/env python3
"""Pre-submit local validation script for 911 Dispatch Supervisor RL Environment."""

from __future__ import annotations

import subprocess
import shutil
import sys
from pathlib import Path


def run_command(
    cmd: list[str], description: str, check: bool = True
) -> subprocess.CompletedProcess:
    print(f"\n{'=' * 60}")
    print(f"CHECK: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        print(f"FAILED: {description}")
        print(f"ERROR: command not found: {cmd[0]}")
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"FAILED: {description}")
        return result
    print(f"PASSED: {description}")
    return result


def _tool_path(name: str) -> str | None:
    """Resolve tool path from PATH or current interpreter's Scripts directory."""
    found = shutil.which(name)
    if found:
        return found

    scripts_dir = Path(sys.executable).resolve().parent
    candidates = [
        scripts_dir / name,
        scripts_dir / f"{name}.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _python_cmd(*args: str) -> list[str]:
    """Build a Python command, preferring uv when available."""
    uv = _tool_path("uv")
    if uv:
        return [uv, "run", "python", *args]
    return [sys.executable, *args]


def check_pytest() -> bool:
    result = run_command(_python_cmd("-m", "pytest", "tests/", "-q"), "All tests pass")
    return result.returncode == 0


def check_inference() -> bool:
    import os

    env = os.environ.copy()
    env["API_BASE_URL"] = "https://api.openai.com/v1"
    env["MODEL_NAME"] = "gpt-4"
    env["OPENAI_API_KEY"] = "dummy-token-for-local-validation"
    env["USE_RANDOM"] = "true"

    print("\nNOTE: Running inference.py in random-agent mode for local validation")
    result = subprocess.run(
        _python_cmd("inference.py"),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    has_start = "[START]" in result.stdout
    has_end = "[END]" in result.stdout

    if has_start and has_end:
        print("PASSED: inference.py produces [START]/[END] output")
        return True
    else:
        print(f"FAILED: inference.py output missing [START] or [END] markers")
        return False


def check_docker_build() -> bool:
    if not shutil.which("docker"):
        print("FAILED: Docker build succeeds")
        print("ERROR: docker command not found")
        return False

    result = run_command(
        ["docker", "build", "-t", "citywide-dispatch-supervisor", "."],
        "Docker build succeeds",
        check=False,
    )
    return result.returncode == 0


def check_openenv_validate() -> bool:
    openenv = _tool_path("openenv")
    if not openenv:
        print("FAILED: openenv validate passes")
        print("ERROR: openenv command not found")
        print("HINT: Install with: pip install openenv-core")
        return False

    result = run_command([openenv, "validate"], "openenv validate passes", check=False)
    return result.returncode == 0


def check_benchmark_scores() -> bool:
    from src.benchmark import list_tasks, run_task

    tasks = list_tasks()
    print(f"\nFound {len(tasks)} tasks:")

    all_valid = True
    for task in tasks:
        task_id = task["task_id"]
        print(f"  - {task_id}: {task['name']} ({task['difficulty']})")

        result = run_task(task_id, seed=42)
        score = result["score"]

        print(f"    Score: {score:.3f}")

        if not (0.0 <= score <= 1.0):
            print(f"    FAILED: Score {score} is outside [0.0, 1.0]")
            all_valid = False
        else:
            print(f"    PASSED: Score is in [0.0, 1.0]")

    return all_valid


def main() -> int:
    print("911 Dispatch RL Environment - Pre-submit Validation")
    print("=" * 60)

    checks = [
        ("pytest", check_pytest),
        ("inference", check_inference),
        ("docker_build", check_docker_build),
        ("openenv_validate", check_openenv_validate),
        ("benchmark_scores", check_benchmark_scores),
    ]

    results: dict[str, bool] = {}

    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            results[name] = False

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ ALL CHECKS PASSED - Ready for submission!")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED - Fix issues before submitting")
        return 1


if __name__ == "__main__":
    sys.exit(main())
