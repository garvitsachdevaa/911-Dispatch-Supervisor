"""Tests for inference.py competition logging format (dispatch domain)."""

from __future__ import annotations

import os
import re
import subprocess
import sys


class TestInferenceFormatCompliance:
    TASK_IDS = ["single_incident", "multi_incident", "mass_casualty", "shift_surge"]

    def _run_inference_capture(self, env: dict[str, str]) -> tuple[int, str, str]:
        cmd = [sys.executable, "inference.py"]
        merged_env = os.environ.copy()
        merged_env.update(env)
        merged_env.setdefault("USE_RANDOM", "true")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=merged_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.returncode, result.stdout, result.stderr

    def test_inference_runs_all_tasks(self) -> None:
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
            "USE_RANDOM": "true",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode == 0, f"inference.py failed: {stderr}"
        tasks_run = []
        for line in stdout.split("\n"):
            if line.startswith("[START]"):
                match = re.match(r"\[START\] task=(\S+) env=(\S+) model=(\S+)", line)
                assert match
                tasks_run.append(match.group(1))
        assert tasks_run == self.TASK_IDS

    def test_start_line_format(self) -> None:
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
            "USE_RANDOM": "true",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = r"\[START\] task=\S+ env=citywide-dispatch-supervisor model=\S+"
        for line in stdout.split("\n"):
            if line.startswith("[START]"):
                assert re.match(pattern, line)

    def test_step_line_error_format(self) -> None:
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
            "USE_RANDOM": "true",
        }
        _, stdout, _ = self._run_inference_capture(env)
        valid_errors = {"null", "max_steps_exceeded", "illegal_transition"}
        for line in stdout.split("\n"):
            if not line.startswith("[STEP]"):
                continue
            match = re.match(r"\[STEP\].+ error=(.+)", line)
            assert match
            assert match.group(1) in valid_errors


class TestEnvVarValidation:
    def _run_inference_capture(self, env: dict[str, str]) -> tuple[int, str, str]:
        cmd = [sys.executable, "inference.py"]
        merged_env = os.environ.copy()
        merged_env.update(env)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=merged_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.returncode, result.stdout, result.stderr

    def test_missing_api_base_url(self) -> None:
        env = {"MODEL_NAME": "m", "HF_TOKEN": "t", "USE_RANDOM": "true"}
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0
        assert "API_BASE_URL" in (stdout + stderr)

    def test_missing_model_name(self) -> None:
        env = {"API_BASE_URL": "x", "HF_TOKEN": "t", "USE_RANDOM": "true"}
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0
        assert "MODEL_NAME" in (stdout + stderr)
