#!/usr/bin/env python3
"""Pre-submit local validation script for ATC Ground Control RL Environment."""

from __future__ import annotations

import subprocess
import sys


def run_command(
    cmd: list[str], description: str, check: bool = True
) -> subprocess.CompletedProcess:
    print(f"\n{'=' * 60}")
    print(f"CHECK: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if check and result.returncode != 0:
        print(f"FAILED: {description}")
        return result
    print(f"PASSED: {description}")
    return result


def check_pytest() -> bool:
    result = run_command(
        ["uv", "run", "python", "-m", "pytest", "tests/", "-q"], "All tests pass"
    )
    return result.returncode == 0


def check_inference() -> bool:
    import os

    env = os.environ.copy()
    env["API_BASE_URL"] = "https://api.openai.com/v1"
    env["MODEL_NAME"] = "gpt-4"
    env["HF_TOKEN"] = "dummy-token-for-local-validation"

    print(
        "\nNOTE: Running inference.py in dummy mode - will emit [START]/[STEP]/[END] lines"
    )
    result = subprocess.run(
        ["uv", "run", "python", "inference.py"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
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
    result = run_command(
        ["docker", "build", "-t", "gradient-ascent-atc", "."],
        "Docker build succeeds",
        check=False,
    )
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
    print("ATC Ground Control RL Environment - Pre-submit Validation")
    print("=" * 60)

    checks = [
        ("pytest", check_pytest),
        ("inference", check_inference),
        ("docker_build", check_docker_build),
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
