"""Tests for inference.py competition logging format."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
from io import StringIO
from typing import Any
from unittest.mock import patch

import pytest


class TestInferenceFormatCompliance:
    """Tests for exact stdout format compliance."""

    TASK_IDS = ["departure", "arrival", "integrated"]

    def _run_inference_capture(
        self, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Run inference.py and capture stdout/stderr."""
        cmd = [sys.executable, "inference.py"]
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=merged_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.returncode, result.stdout, result.stderr

    def _parse_output(self, stdout: str) -> dict[str, list[dict[str, Any]]]:
        """Parse stdout into structured log entries."""
        lines = stdout.strip().split("\n")
        parsed = {"start": [], "step": [], "end": []}
        for line in lines:
            if line.startswith("[START]"):
                match = re.match(r"\[START\] task=(\S+) env=(\S+) model=(\S+)", line)
                if match:
                    parsed["start"].append(
                        {
                            "task": match.group(1),
                            "env": match.group(2),
                            "model": match.group(3),
                        }
                    )
            elif line.startswith("[STEP]"):
                match = re.match(
                    r"\[STEP\] step=(\d+) action=(.+) reward=(\d+\.\d{2}) done=(true|false) error=(.+)",
                    line,
                )
                if match:
                    parsed["step"].append(
                        {
                            "step": int(match.group(1)),
                            "action": match.group(2),
                            "reward": match.group(3),
                            "done": match.group(4),
                            "error": match.group(5),
                        }
                    )
            elif line.startswith("[END]"):
                match = re.match(
                    r"\[END\] success=(true|false) steps=(\d+) score=(\d+\.\d{3}) rewards=(.+)",
                    line,
                )
                if match:
                    parsed["end"].append(
                        {
                            "success": match.group(1),
                            "steps": int(match.group(2)),
                            "score": match.group(3),
                            "rewards": match.group(4),
                        }
                    )
        return parsed

    def test_inference_runs_all_three_tasks(self) -> None:
        """Inference must run all three task IDs sequentially."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode == 0, f"inference.py failed: {stderr}"
        parsed = self._parse_output(stdout)
        tasks_run = [s["task"] for s in parsed["start"]]
        assert tasks_run == self.TASK_IDS, f"Expected {self.TASK_IDS}, got {tasks_run}"

    def test_start_line_format(self) -> None:
        """START line must match exact format: [START] task=X env=gradient-ascent-atc model=Y"""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "gpt-4o",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = r"\[START\] task=\S+ env=gradient-ascent-atc model=\S+"
        for line in stdout.split("\n"):
            if line.startswith("[START]"):
                assert re.match(pattern, line), f"START line format invalid: {line}"

    def test_step_line_reward_format(self) -> None:
        """STEP line reward must be exactly 2 decimal places."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = r"\[STEP\] step=\d+ action=.+ reward=(\d+\.\d{2}) done=(true|false) error=.+"
        for line in stdout.split("\n"):
            if line.startswith("[STEP]"):
                match = re.match(pattern, line)
                assert match, f"STEP line format invalid: {line}"
                reward = match.group(1)
                assert re.match(r"\d+\.\d{2}", reward), f"Reward format wrong: {reward}"

    def test_step_line_done_format(self) -> None:
        """STEP line done must be lowercase true or false."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = (
            r"\[STEP\] step=\d+ action=.+ reward=\d+\.\d{2} done=(true|false) error=.+"
        )
        for line in stdout.split("\n"):
            if line.startswith("[STEP]"):
                match = re.match(pattern, line)
                assert match, f"STEP line format invalid: {line}"
                done = match.group(1)
                assert done in ("true", "false"), f"Done must be true/false: {done}"

    def test_step_line_error_null_format(self) -> None:
        """STEP line error must be null (not None or empty) for normal steps."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = r"\[STEP\] step=\d+ action=.+ reward=\d+\.\d{2} done=(true|false) error=(.+)"
        valid_errors = {"null", "max_steps_exceeded", "illegal_transition"}
        for line in stdout.split("\n"):
            if line.startswith("[STEP]"):
                match = re.match(pattern, line)
                assert match, f"STEP line format invalid: {line}"
                error = match.group(2)
                assert error in valid_errors, (
                    f"Error must be one of {valid_errors}: {error}"
                )

    def test_end_line_score_format(self) -> None:
        """END line score must be exactly 3 decimal places."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = (
            r"\[END\] success=(true|false) steps=\d+ score=(\d+\.\d{3}) rewards=.+"
        )
        for line in stdout.split("\n"):
            if line.startswith("[END]"):
                match = re.match(pattern, line)
                assert match, f"END line format invalid: {line}"
                score = match.group(2)
                assert re.match(r"\d+\.\d{3}", score), f"Score format wrong: {score}"

    def test_end_line_success_format(self) -> None:
        """END line success must be lowercase true or false."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = r"\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=.+"
        for line in stdout.split("\n"):
            if line.startswith("[END]"):
                match = re.match(pattern, line)
                assert match, f"END line format invalid: {line}"
                success = match.group(1)
                assert success in ("true", "false"), (
                    f"Success must be true/false: {success}"
                )

    def test_end_line_rewards_format(self) -> None:
        """END line rewards must be comma-separated with 2 decimal places each."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        _, stdout, _ = self._run_inference_capture(env)
        pattern = (
            r"\[END\] success=(true|false) steps=\d+ score=\d+\.\d{3} rewards=(.+)"
        )
        for line in stdout.split("\n"):
            if line.startswith("[END]"):
                match = re.match(pattern, line)
                assert match, f"END line format invalid: {line}"
                rewards_str = match.group(2)
                rewards = rewards_str.split(",")
                for r in rewards:
                    assert re.match(r"\d+\.\d{2}", r), (
                        f"Reward format wrong in list: {r}"
                    )


class TestEnvVarValidation:
    """Tests for environment variable validation."""

    def _run_inference_capture(
        self, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Run inference.py and capture stdout/stderr."""
        cmd = [sys.executable, "inference.py"]
        merged_env = os.environ.copy()
        if env:
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
        """Must fail with clear error when API_BASE_URL is missing."""
        env = {
            "MODEL_NAME": "test-model",
            "HF_TOKEN": "test-token",
        }
        env = {k: v for k, v in env.items() if k != "API_BASE_URL"}
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0, "Should fail when API_BASE_URL is missing"
        error_output = stdout + stderr
        assert "API_BASE_URL" in error_output, (
            f"Error should mention API_BASE_URL: {error_output}"
        )

    def test_missing_model_name(self) -> None:
        """Must fail with clear error when MODEL_NAME is missing."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "HF_TOKEN": "test-token",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0, "Should fail when MODEL_NAME is missing"
        error_output = stdout + stderr
        assert "MODEL_NAME" in error_output, (
            f"Error should mention MODEL_NAME: {error_output}"
        )

    def test_missing_hf_token(self) -> None:
        """Must fail with clear error when HF_TOKEN is missing."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0, "Should fail when HF_TOKEN is missing"
        error_output = stdout + stderr
        assert "HF_TOKEN" in error_output, (
            f"Error should mention HF_TOKEN: {error_output}"
        )


class TestFailurePathEndEmission:
    """Tests for guaranteed [END] emission on failure paths."""

    def _run_inference_capture(
        self, env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Run inference.py and capture stdout/stderr."""
        cmd = [sys.executable, "inference.py"]
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=merged_env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        return result.returncode, result.stdout, result.stderr

    def test_end_emitted_on_missing_env_vars(self) -> None:
        """Must emit [END] line even when env vars are missing."""
        env = {
            "API_BASE_URL": "https://api.example.com",
            "MODEL_NAME": "test-model",
        }
        returncode, stdout, stderr = self._run_inference_capture(env)
        assert returncode != 0, "Should fail when HF_TOKEN is missing"
        assert "[END]" in stdout, "Must emit [END] even on missing env var failure"


class TestAsyncResetUsage:
    """Tests verifying async/await is used for env.reset()."""

    def test_no_sync_reset_in_code(self) -> None:
        """Source code must not contain synchronous env.reset() calls."""
        with open("inference.py", "r") as f:
            content = f.read()
        # Match env.reset() NOT preceded by await (negative lookbehind)
        sync_reset_pattern = r"(?<!await\s)env\.reset\(\s*\)"
        matches = re.findall(sync_reset_pattern, content)
        assert len(matches) == 0, f"Found synchronous env.reset() call(s): {matches}"
        assert "await env.reset()" in content, (
            "Must have 'await env.reset()' for async reset"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
