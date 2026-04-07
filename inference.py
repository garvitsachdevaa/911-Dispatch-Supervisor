"""Competition inference script with exact stdout logging format."""

import asyncio
import json
import os
import random
import sys
import time
from typing import Any

from openai import OpenAI

from src.models import Action, DispatchAction
from src.openenv_environment import OpenEnvEnvironment

# ---------------------------------------------------------------------------
# Action 2 — Canonical environment variable names + OpenAI client
# ---------------------------------------------------------------------------
# HuggingFace deprecated https://api-inference.huggingface.co/v1 (HTTP 410)
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
API_KEY      = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# Action 4 — Per-task max-steps (must match the environment fixtures)
# ---------------------------------------------------------------------------
TASK_MAX_STEPS: dict[str, int] = {
    "single_incident": 20,
    "multi_incident": 40,
    "mass_casualty": 60,
    "shift_surge": 60,
}

# ---------------------------------------------------------------------------
# Action 1 — JSON structured logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action, reward: float, done: bool, error=None):
    done_str = "true" if done else "false"
    err_str = "null" if error is None else str(error)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class RandomAgent:
    """Deterministic baseline agent that picks legal actions at random."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def select_action(
        self, legal_actions: list[Action], state_desc: str = "", prev_obs: Any = None
    ) -> Action | None:
        """Select a random legal action.

        Args:
            legal_actions: List of valid actions for current state.

        Returns:
            Selected Action, or None if no legal actions available.
        """
        if not legal_actions:
            return None
        return self._rng.choice(legal_actions)


class LLMAgent:
    """LLM agent using OpenAI-compatible endpoint (sync client)."""

    def __init__(self) -> None:
        # Action 2 — single canonical OpenAI client
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        self._rng = random.Random(42)

    def _call_llm_sync(self, messages: list[dict]) -> str:
        """Blocking LLM call — must be run in a thread to avoid blocking the event loop."""
        try:
            resp = self.client.chat.completions.create(
                model=MODEL_NAME, messages=messages
            )
            return resp.choices[0].message.content or ""
        except Exception:
            return ""

    async def select_action(
        self, legal_actions: list[Action], state_desc: str, prev_obs: Any = None
    ) -> Action | None:
        """Select an action via LLM (async — offloads blocking HTTP to a thread)."""
        if not legal_actions:
            return None

        SYSTEM_PROMPT = """You are an expert 911 dispatch supervisor for Metro City.

STRICT PRIORITY ORDER:
1. P1 incidents (cardiac arrest, shooting, building collapse) = dispatch IMMEDIATELY. Any P1 death caps your score at 0.2.
2. Match unit type exactly: MEDIC→medical emergencies, ENGINE/LADDER→fire, PATROL→crime/shooting, HAZMAT→hazmat.
3. Never dispatch a unit already DISPATCHED or ON_SCENE.
4. Use mutual_aid ONLY when ALL local units of the needed type are busy.
5. Use stage to pre-position units near high-risk areas when no active incidents need them.

SCORING WEIGHTS: response_time 30% | triage 25% | P1 survival 25% | coverage 12% | protocol 8%

You will receive current state and a numbered list of legal actions.
Respond with ONLY the exact action string. No explanation. No JSON. Just the string."""

        prev_info = ""
        if prev_obs and hasattr(prev_obs, "issues") and prev_obs.issues:
            prev_info = f"\nPrevious action issues: {', '.join(prev_obs.issues)}. Adapt your next action accordingly."

        action_strs = [f"- {_format_action(a)}" for a in legal_actions]
        user_prompt = (
            f"Current state: {state_desc}{prev_info}\n\nLegal actions:\n"
            + "\n".join(action_strs)
            + "\n\nRespond with ONLY the correct action string."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Run the blocking sync OpenAI call in a thread pool so it doesn't
        # block the asyncio event loop (which owns env.reset / env.step).
        response = await asyncio.to_thread(self._call_llm_sync, messages)

        if not response:
            return self._rng.choice(legal_actions)

        response_norm = response.strip().lower()
        for action in legal_actions:
            if _format_action(action).lower() == response_norm:
                return action

        # Fallback to random if LLM response doesn't match any legal action
        return self._rng.choice(legal_actions)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_action(action: Action) -> str:
    """Format dispatch action as a compact string for logging and LLM matching."""
    base = f"{action.action_type.value} {action.unit_id}->{action.incident_id}"
    if action.priority_override is not None:
        base += f" prio={action.priority_override.value}"
    return base


def _format_state_for_llm(env: OpenEnvEnvironment) -> str:
    state = env.state()
    available_units = [u.unit_id for u in state.units.values() if u.status.value == "AVAILABLE"]
    active_incidents = [
        i
        for i in state.incidents.values()
        if i.status.value not in {"RESOLVED"}
    ]

    parts: list[str] = []
    parts.append(f"city_time={state.city_time:.0f}s step={state.step_count}")
    parts.append(f"available_units={len(available_units)}")
    parts.append(f"active_incidents={len(active_incidents)}")

    if active_incidents:
        brief = ", ".join(
            f"{i.incident_id}({i.incident_type.value},{i.severity.value},{i.status.value})"
            for i in sorted(active_incidents, key=lambda x: x.incident_id)[:6]
        )
        parts.append(f"incidents=[{brief}]")

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(
    task_id: str,
    agent: RandomAgent | LLMAgent,
) -> tuple[bool, int, float, list[float]]:
    """Run a single episode for a task.

    Returns:
        Tuple of (success, step_count, score, list_of_rewards).
    """
    # Action 4 — per-task max steps
    max_steps = TASK_MAX_STEPS.get(task_id, 60)

    # Action 3 — MAX_TOTAL_REWARD for score normalization
    MAX_TOTAL_REWARD = max_steps * 1.0

    # Action 1 — log_start before the episode loop
    log_start(task=task_id, env="citywide-dispatch-supervisor", model=MODEL_NAME)

    env = OpenEnvEnvironment(task_id=task_id, seed=42)
    step_count = 0
    rewards: list[float] = []
    success = False
    score = 0.0

    try:
        observation = await env.reset()
        prev_obs = observation

        while step_count < max_steps:
            step_count += 1

            legal_actions = env.legal_actions()
            state_desc = _format_state_for_llm(env)

            # LLMAgent.select_action is async; RandomAgent's is sync — handle both
            if isinstance(agent, LLMAgent):
                action = await agent.select_action(legal_actions, state_desc, prev_obs)
            else:
                action = agent.select_action(legal_actions, state_desc, prev_obs)

            if action is None:
                # No legal actions — end episode
                log_step(step=step_count, action="NONE", reward=0.0, done=True, error=None)
                break

            try:
                obs, reward, done = await env.step(action)
                prev_obs = obs
                rewards.append(reward)

                # Terminal conditions
                has_illegal_transition = any(
                    ("illegal" in issue) for issue in (obs.issues or [])
                )

                if done or has_illegal_transition:
                    err = "illegal_transition" if has_illegal_transition else None
                    if has_illegal_transition:
                        success = False
                    log_step(
                        step=step_count,
                        action=_format_action(action),
                        reward=reward,
                        done=True,
                        error=err,
                    )
                    break

                # Normal step log
                log_step(
                    step=step_count,
                    action=_format_action(action),
                    reward=reward,
                    done=False,
                    error=None,
                )

            except Exception as e:
                log_step(
                    step=step_count,
                    action=_format_action(action),
                    reward=0.0,
                    done=True,
                    error=e,
                )
                success = False
                break

        # ------------------------------------------------------------------
        # Action 3 — Score computation
        # ------------------------------------------------------------------
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= 0.5

    except Exception:
        score = 0.0
        success = False
    finally:
        env.close()
        # Action 1 — log_end in the finally block
        log_end(success=success, steps=step_count, score=round(score, 4), rewards=rewards)

    return success, step_count, score, rewards


# ---------------------------------------------------------------------------
# Main — runs all 4 tasks sequentially (Action 4)
# ---------------------------------------------------------------------------

async def main() -> int:
    """Main entry point for inference script."""
    use_random = os.environ.get("USE_RANDOM", "").lower() == "true"

    if use_random:
        agent: RandomAgent | LLMAgent = RandomAgent(seed=42)
    else:
        if not API_KEY:
            log_end(success=False, steps=0, score=0.0, rewards=[])
            print("ERROR: Missing HF_TOKEN environment variable", file=sys.stderr)
            return 1
        agent = LLMAgent()

    task_ids = ["single_incident", "multi_incident", "mass_casualty", "shift_surge"]

    # Action 4 — wall-clock timing per task
    total_start = time.time()

    for task_id in task_ids:
        task_start = time.time()
        await run_episode(task_id, agent)
        task_elapsed = time.time() - task_start
        print(
            f"[TIMER] task={task_id} elapsed={task_elapsed:.1f}s",
            file=sys.stderr,
        )

    total_elapsed = time.time() - total_start
    print(f"[TIMER] total_elapsed={total_elapsed:.1f}s", file=sys.stderr)

    if total_elapsed > 900:  # 15-minute budget
        print(
            "[WARNING] Total inference time exceeded 15 minutes! "
            "Consider reducing LLM retries or adding a sleep cap.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
