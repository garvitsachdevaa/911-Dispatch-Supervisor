"""Smoke test for Docker image health and reset endpoints."""

import subprocess
import time
import requests  # type: ignore


IMAGE_NAME = "gradient-ascent-atc"
CONTAINER_NAME = "test-atc-smoke"
HOST = "http://localhost:8000"


def build_image() -> None:
    print("Building Docker image...")
    result = subprocess.run(
        ["docker", "build", "-t", IMAGE_NAME, "."],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Docker build failed:\n{result.stderr}")
    print("Build succeeded.")


def run_container() -> None:
    print("Running container...")
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "-p",
            "8000:8000",
            "--name",
            CONTAINER_NAME,
            IMAGE_NAME,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    # Give the server time to start
    time.sleep(5)
    print("Container running.")


def kill_container() -> None:
    print("Killing container...")
    subprocess.run(["docker", "kill", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", CONTAINER_NAME], capture_output=True)
    print("Container cleaned up.")


def test_health() -> None:
    print("Testing /health endpoint...")
    response = requests.get(f"{HOST}/health", timeout=10)
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert response.json() == {"status": "ok"}, f"Unexpected body: {response.json()}"
    print("Health check passed.")


def test_reset() -> None:
    print("Testing /reset endpoint...")
    response = requests.post(
        f"{HOST}/reset",
        json={"task_id": "arrival", "seed": 42},
        timeout=10,
    )
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print("Reset check passed.")


def main() -> None:
    try:
        build_image()
        run_container()
        test_health()
        test_reset()
        print("\nAll smoke tests passed!")
    finally:
        kill_container()


if __name__ == "__main__":
    main()
