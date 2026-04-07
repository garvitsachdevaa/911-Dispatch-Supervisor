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
            "OPENAI_API_KEY": "test-token",
            "USE_RANDOM": "true",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode == 0, f"inference.py failed: {stderr}"
        tasks_run = []
        for line in stdout.split("\n"):
            if '"type": "START"' in line:
                try:
                    import json
                    d = json.loads(line)
                    tasks_run.append(d.get("task"))
                except:
                    pass
        assert tasks_run == self.TASK_IDS

    def test_start_line_format(self) -> None:
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "OPENAI_API_KEY": "test-token",
            "USE_RANDOM": "true",
        }
        _, stdout, _ = self._run_inference_capture(env)
        for line in stdout.split("\n"):
            if '"type": "START"' in line:
                import json
                d = json.loads(line)
                assert d.get("task") in self.TASK_IDS
                assert d.get("env") == "citywide-dispatch-supervisor"
                assert d.get("model") == "test-model"

    def test_step_line_error_format(self) -> None:
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "OPENAI_API_KEY": "test-token",
            "USE_RANDOM": "true",
        }
        _, stdout, _ = self._run_inference_capture(env)
        valid_errors = {None, "max_steps_exceeded", "illegal_transition", "step_error"}
        for line in stdout.split("\n"):
            if '"type": "STEP"' in line:
                import json
                d = json.loads(line)
                assert d.get("error") in valid_errors or isinstance(d.get("error"), str)


class TestEnvVarValidation:
    def _run_inference_capture(self, env: dict[str, str]) -> tuple[int, str, str]:
        cmd = [sys.executable, "inference.py"]
        merged_env = os.environ.copy()
        merged_env.update(env)

        # Ensure tests are not affected by host environment variables.
        # If the test doesn't provide a required var, explicitly remove it.
        if "API_BASE_URL" not in env:
            merged_env.pop("API_BASE_URL", None)
        if "MODEL_NAME" not in env:
            merged_env.pop("MODEL_NAME", None)
        if "OPENAI_API_KEY" not in env:
            merged_env.pop("OPENAI_API_KEY", None)
        if "HF_TOKEN" not in env:
            merged_env.pop("HF_TOKEN", None)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=merged_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.returncode, result.stdout, result.stderr

    def test_missing_api_key_when_not_random(self) -> None:
        env = {
            "USE_RANDOM": "false",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0
        assert "HF_TOKEN" in (stdout + stderr)
