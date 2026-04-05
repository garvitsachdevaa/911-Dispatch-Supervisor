"""Reward engine and grader primitives."""

from pydantic import BaseModel, Field

from src.models import (
    Action,
    DispatchAction,
    IncidentSeverity,
    Observation,
    State,
    UnitStatus,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class RewardSignal(BaseModel):
    """Signal components for reward breakdown."""

    model_config = {"extra": "forbid"}

    response_time: float = Field(..., ge=0.0, le=1.0)
    triage: float = Field(..., ge=0.0, le=1.0)
    survival: float = Field(..., ge=0.0, le=1.0)
    coverage: float = Field(..., ge=0.0, le=1.0)
    protocol: float = Field(..., ge=0.0, le=1.0)


class RewardCalculator:
    """Evaluates dispatcher decisions with response-time, triage, survival, coverage, protocol."""

    weights: dict[str, float] = {
        "response_time": 0.30,
        "triage": 0.25,
        "survival": 0.25,
        "coverage": 0.12,
        "protocol": 0.08,
    }

    def compute_reward(self, state: State, action: Action, obs: Observation) -> tuple[RewardSignal, float]:
        """Compute reward signal and total weighted score.

        Args:
            state: Current lifecycle state
            action: Action taken by agent
            obs: Observation returned by environment

        Returns:
            Tuple of (reward signal components, total weighted score clamped to [0.0, 1.0])
        """
        response_time = self._compute_response_time(state, action)
        triage = self._compute_triage(state, action)
        survival = self._compute_survival(state)
        coverage = self._compute_coverage(state)
        protocol = self._compute_protocol(obs)

        signal = RewardSignal(
            response_time=response_time,
            triage=triage,
            survival=survival,
            coverage=coverage,
            protocol=protocol,
        )

        total = self._compute_weighted_total(signal, state)

        return signal, total

    def _compute_response_time(self, state: State, action: Action) -> float:
        """Score dispatch timeliness via ETA benchmarks.

        If no dispatch occurs this step, return a neutral 0.5.
        """
        if action.action_type != DispatchAction.DISPATCH:
            return 0.5

        unit = state.units.get(action.unit_id)
        incident = state.incidents.get(action.incident_id)
        if unit is None or incident is None:
            return 0.0

        benchmark: float
        if incident.severity == IncidentSeverity.PRIORITY_1:
            benchmark = 240.0
        elif incident.severity == IncidentSeverity.PRIORITY_2:
            benchmark = 480.0
        else:
            benchmark = 900.0

        eta = max(float(unit.eta_seconds), 1e-6)
        return _clamp01(benchmark / eta)

    def _compute_triage(self, state: State, action: Action) -> float:
        """Score whether dispatched unit type matches the incident's required types."""
        if action.action_type != DispatchAction.DISPATCH:
            return 0.5

        unit = state.units.get(action.unit_id)
        incident = state.incidents.get(action.incident_id)
        if unit is None or incident is None:
            return 0.0

        required_map = state.metadata.get("default_required_units", {})
        required_types = required_map.get(str(incident.incident_type), [])
        if not required_types:
            return 0.5

        # required_types are stored as strings in metadata.
        if str(unit.unit_type) in set(required_types):
            return 1.0
        return 0.0

    def _compute_survival(self, state: State) -> float:
        """Score survival outcomes for Priority-1 incidents.

        Uses state.metadata bookkeeping written by the state machine.
        """
        p1_seen: list[str] = list(state.metadata.get("p1_seen", []))
        if not p1_seen:
            return 1.0

        resolved: set[str] = set(state.metadata.get("resolved_incidents", []))
        failed: set[str] = set(state.metadata.get("failed_incidents", []))

        ok = 0
        for incident_id in p1_seen:
            if incident_id in resolved and incident_id not in failed:
                ok += 1
        return _clamp01(ok / max(len(p1_seen), 1))

    def _compute_coverage(self, state: State) -> float:
        """Score geographic coverage of AVAILABLE units across districts.

        Districts are derived by slicing the x-axis into equal bins.
        """
        districts: list[str] = list(state.metadata.get("districts", []))
        grid_size = state.metadata.get("grid_size")

        if not districts or not grid_size:
            return 1.0

        width = float(grid_size[0])
        if width <= 0.0:
            return 1.0

        covered: set[int] = set()
        bin_width = width / len(districts)
        for unit in state.units.values():
            if unit.status != UnitStatus.AVAILABLE:
                continue
            idx = int(min(len(districts) - 1, max(0.0, unit.location_x) // max(bin_width, 1e-6)))
            covered.add(idx)

        return _clamp01(len(covered) / len(districts))

    def _compute_protocol(self, obs: Observation) -> float:
        return 1.0 if obs.protocol_ok else 0.0

    def _compute_weighted_total(self, signal: RewardSignal, state: State) -> float:
        total = (
            signal.response_time * self.weights["response_time"]
            + signal.triage * self.weights["triage"]
            + signal.survival * self.weights["survival"]
            + signal.coverage * self.weights["coverage"]
            + signal.protocol * self.weights["protocol"]
        )

        total = _clamp01(total)

        # Dominance rule: if any Priority-1 incidents existed and survival == 0.0, cap score.
        if state.metadata.get("p1_seen") and signal.survival == 0.0:
            total = min(total, 0.2)

        return total


class TaskGrader:
    """Aggregates episode rewards and returns final normalized score."""

    def grade_episode(self, episode_rewards: list[float], task_id: str) -> float:
        """Aggregate rewards over episode and return final score.

        Args:
            episode_rewards: List of per-step reward values
            task_id: Task identifier (unused in base grader)

        Returns:
            Final score in [0.0, 1.0]
        """
        if not episode_rewards:
            return 0.0

        total = sum(episode_rewards)
        avg = total / len(episode_rewards)

        return max(0.0, min(1.0, avg))
