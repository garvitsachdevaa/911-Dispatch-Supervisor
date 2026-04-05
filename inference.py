"""Competition inference script with exact stdout logging format."""

import asyncio
import os
import random
import sys
from typing import Any

import httpx

from src.models import Action, ClearanceType
from src.openenv_environment import OpenEnvEnvironment


def _validate_env_vars() -> None:
    """Validate required environment variables are set."""
    missing = []
    for var in ("API_BASE_URL", "MODEL_NAME"):
        if not os.environ.get(var):
            missing.append(var)
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    use_random = os.environ.get("USE_RANDOM", "").lower() == "true"
    api_base_url = os.environ.get("API_BASE_URL", "")
    is_gemini = "gemini" in api_base_url.lower()
    if not use_random and not is_gemini and not os.environ.get("HF_TOKEN"):
        raise EnvironmentError("Missing required environment variable: HF_TOKEN")


def _get_env(var: str) -> str:
    """Get environment variable value."""
    return os.environ[var]


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
            Selected Action, or None if no legal actions available (agent should wait).
        """
        if not legal_actions:
            return None
        return self._rng.choice(legal_actions)


class LLMAgent:
    """LLM agent using OpenAI-compatible endpoint with dynamic auth."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def chat(self, messages: list[dict]) -> str:
        """Send chat request to LLM endpoint with appropriate auth.

        Auth method depends on endpoint:
        - Gemini (contains 'gemini'): use ?key= query param
        - Groq (contains 'groq'): use Authorization: Bearer header
        - Other OpenAI-compatible: use Authorization: Bearer header
        """
        is_gemini = "gemini" in self.base_url.lower()
        headers = {"Content-Type": "application/json"}

        if is_gemini:
            url = f"{self.base_url}/chat/completions?key={self.api_key}"
        else:
            url = f"{self.base_url}/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                url, json={"model": self.model, "messages": messages}, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def select_action(
        self, legal_actions: list[Action], state_desc: str, prev_obs: Any = None
    ) -> Action | None:
        if not legal_actions:
            return None

        SYSTEM_PROMPT = """You are an expert ATC ground control agent. Aircraft lifecycle phases (in order):

ARRIVAL: approach -> landing -> arrival_handoff -> taxi_in -> docking -> at_gate
DEPARTURE: at_gate -> pushback -> taxi_out -> departure_queue -> takeoff -> departed

Rules:
- at_gate: After ~60s turnaround, MUST issue PUSHBACK first (never taxi before pushback)
- pushback: issue TAXI with route to departure queue (e.g. taxiway A, taxiway B, departure_queue)
- taxi_out: issue HOLD_SHORT at runway holding point before crossing active runway
- departure_queue: wait, then issue LINE_UP when runway is clear for immediate departure
- approach: issue LANDING clearance when aircraft is on final approach
- arrival_handoff: issue TAXI with route to assigned gate (e.g. runway_exit, taxiway C, gate)
- taxi_in: issue HOLD_SHORT before crossing any active runway

Respond with ONLY the exact action string from the legal actions list. No explanation."""

        prev_info = ""
        if prev_obs and prev_obs.issues:
            prev_info = f"\nPrevious action issues: {', '.join(prev_obs.issues)}. Adapt your next action accordingly."

        action_strs = [
            f"- {a.clearance_type.value} {a.target_callsign}" for a in legal_actions
        ]
        user_prompt = (
            f"Current state: {state_desc}{prev_info}\n\nLegal actions:\n"
            + "\n".join(action_strs)
            + "\n\nRespond with ONLY the correct action string."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = await self.chat(messages)

        # Parse response and match to legal action
        for action in legal_actions:
            action_repr = f"{action.clearance_type.value} {action.target_callsign}"
            if action_repr.lower() in response.lower():
                return action

        # Fallback to random if LLM response doesn't match
        return random.choice(legal_actions)


def _format_action(action: Action) -> str:
    """Format action as a compact string for logging.

    Args:
        action: Action to format.

    Returns:
        Compact string representation of the action.
    """
    parts = [action.clearance_type.value, action.target_callsign]
    if action.route:
        parts.append(",".join(action.route[:2]))
    return "|".join(parts)


async def run_episode(
    task_id: str,
    model_name: str,
    agent: RandomAgent | LLMAgent,
) -> tuple[bool, int, float, list[float]]:
    """Run a single episode for a task.

    Args:
        task_id: Task identifier (departure, arrival, integrated).
        model_name: Model name for logging.
        agent: Agent to use for action selection (RandomAgent or LLMAgent).

    Returns:
        Tuple of (success, step_count, total_score, list_of_rewards).
    """
    print(f"[START] task={task_id} env=gradient-ascent-atc model={model_name}")

    env = OpenEnvEnvironment(task_id=task_id, seed=42)
    step_count = 0
    rewards: list[float] = []
    success = False
    error_msg: str | None = None

    try:
        observation = await env.reset()
        rewards.append(observation.score)
        prev_obs = observation

        while True:
            step_count += 1

            state = env.state()
            machine = env._machine
            legal_actions = machine.get_legal_actions(machine._state)

            if isinstance(agent, LLMAgent):
                state_desc = (
                    f"phase={state.phase}, aircraft={list(state.aircraft.keys())}"
                )
                action = await agent.select_action(legal_actions, state_desc, prev_obs)
            else:
                action = agent.select_action(legal_actions)

            if action is None:
                # No legal actions - create a no-op action to continue
                action = Action(
                    clearance_type=ClearanceType.TAXI,
                    target_callsign="BAW123",
                    route=[],
                )
            action_str = _format_action(action)

            try:
                obs, reward, done = await env.step(action)
                prev_obs = obs
                rewards.append(reward)

                # Check for terminal conditions
                machine = env._machine
                is_terminal = machine.is_terminal(machine._state)
                has_illegal_transition = "illegal_transition" in obs.issues

                # Terminal conditions: done flag OR terminal state OR illegal transition
                if done or is_terminal or has_illegal_transition:
                    if has_illegal_transition:
                        error_str = "illegal_transition"
                        success = False
                    else:
                        error_str = "null"
                        success = True
                    reward_str = f"{reward:.2f}"
                    print(
                        f"[STEP] step={step_count} action={action_str} "
                        f"reward={reward_str} done=true error={error_str}"
                    )
                    break

                # Format reward with exactly 2 decimal places
                reward_str = f"{reward:.2f}"

                # Format error - use null (not None)
                error_str = "null"

                print(
                    f"[STEP] step={step_count} action={action_str} "
                    f"reward={reward_str} done={str(done).lower()} error={error_str}"
                )

                # Safety check for runaway episodes
                if step_count >= 1000:
                    error_msg = "max_steps_exceeded"
                    print(
                        f"[STEP] step={step_count} action={action_str} "
                        f"reward={reward_str} done=true error=max_steps_exceeded"
                    )
                    success = False
                    break

            except Exception as e:
                error_msg = str(e)
                print(
                    f"[STEP] step={step_count} action={action_str} "
                    f"reward=0.00 done=true error={error_msg}"
                )
                success = False
                break

    except Exception as e:
        error_msg = str(e)
        success = False
    finally:
        env.close()

    # Calculate total score (average of rewards)
    total_score = sum(rewards) / len(rewards) if rewards else 0.0
    total_score = max(0.0, min(1.0, total_score))  # Clamp to [0.0, 1.0]

    # Format rewards list as comma-separated with 2 decimal places
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={step_count} "
        f"score={total_score:.3f} rewards={rewards_str}"
    )

    return success, step_count, total_score, rewards


async def main() -> int:
    """Main entry point for inference script."""
    try:
        _validate_env_vars()

        model_name = _get_env("MODEL_NAME")
        api_base_url = _get_env("API_BASE_URL")

        use_random = os.environ.get("USE_RANDOM", "").lower() == "true"

        if use_random:
            agent: RandomAgent | LLMAgent = RandomAgent(seed=42)
        else:
            hf_token = os.environ.get("HF_TOKEN", "")
            agent = LLMAgent(api_key=hf_token, base_url=api_base_url, model=model_name)

        task_ids = ["departure", "arrival", "integrated"]

        for task_id in task_ids:
            await run_episode(task_id, model_name, agent)

        return 0

    except EnvironmentError as e:
        # Emit [END] to stdout for failure case
        print("[END] success=false steps=0 score=0.000 rewards=")
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        # Emit [END] to stdout for failure case
        print("[END] success=false steps=0 score=0.000 rewards=")
        print(f"ERROR: Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
