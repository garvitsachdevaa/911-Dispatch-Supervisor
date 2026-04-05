"""Tests for reward engine and grader primitives."""

import pytest

from src.models import Action, ClearanceType, LifecyclePhase, Observation
from src.rewards import RewardCalculator, RewardSignal, TaskGrader
from src.state_machine import LifecycleState


class TestRewardSignal:
    """Tests for RewardSignal Pydantic model."""

    def test_reward_signal_creation(self) -> None:
        """Test creating a RewardSignal with valid values."""
        signal = RewardSignal(
            safety=1.0,
            legality=1.0,
            completion=0.5,
            efficiency=0.8,
            communication=0.9,
        )
        assert signal.safety == 1.0
        assert signal.legality == 1.0
        assert signal.completion == 0.5
        assert signal.efficiency == 0.8
        assert signal.communication == 0.9

    def test_reward_signal_all_zeros(self) -> None:
        """Test creating a RewardSignal with all zeros."""
        signal = RewardSignal(
            safety=0.0,
            legality=0.0,
            completion=0.0,
            efficiency=0.0,
            communication=0.0,
        )
        assert signal.safety == 0.0

    def test_reward_signal_extra_fields_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(Exception):
            RewardSignal(
                safety=1.0,
                legality=1.0,
                completion=0.5,
                efficiency=0.8,
                communication=0.9,
                extra_field=0.5,  # Should be rejected
            )

    def test_reward_signal_default_values_not_allowed(self) -> None:
        """Test that all fields must be explicitly provided."""
        with pytest.raises(Exception):
            RewardSignal()  # All fields required


class TestRewardCalculator:
    """Tests for RewardCalculator."""

    @pytest.fixture
    def calculator(self) -> RewardCalculator:
        """Create a RewardCalculator instance."""
        return RewardCalculator()

    @pytest.fixture
    def safe_state(self) -> LifecycleState:
        """Create a safe lifecycle state (no violations)."""
        return LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="test-123",
            step_count=0,
            task_id="arrival",
            completed_phases=[],
            metadata={},
        )

    @pytest.fixture
    def safe_action(self) -> Action:
        """Create a safe valid action."""
        return Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27L",
        )

    @pytest.fixture
    def safe_obs(self) -> Observation:
        """Create a safe observation."""
        return Observation(
            result="clear",
            score=1.0,
            phraseology_ok=True,
            issues=[],
        )

    def test_compute_reward_returns_tuple(
        self,
        calculator: RewardCalculator,
        safe_state: LifecycleState,
        safe_action: Action,
        safe_obs: Observation,
    ) -> None:
        """Test that compute_reward returns (RewardSignal, float)."""
        result = calculator.compute_reward(safe_state, safe_action, safe_obs)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], RewardSignal)
        assert isinstance(result[1], float)

    def test_total_score_in_range(
        self,
        calculator: RewardCalculator,
        safe_state: LifecycleState,
        safe_action: Action,
        safe_obs: Observation,
    ) -> None:
        """Test that total score is clamped to [0.0, 1.0]."""
        signal, total = calculator.compute_reward(safe_state, safe_action, safe_obs)
        assert 0.0 <= total <= 1.0

    def test_safety_dominance_zero_safety(
        self,
        calculator: RewardCalculator,
        safe_state: LifecycleState,
        safe_action: Action,
    ) -> None:
        """Test that safety=0 forces total=0 regardless of other scores."""
        # Observation with collision/runway incursion issue
        obs = Observation(
            result="collision",
            score=0.0,
            phraseology_ok=False,
            issues=["collision"],
        )
        signal, total = calculator.compute_reward(safe_state, safe_action, obs)
        assert signal.safety == 0.0
        assert total == 0.0  # Safety dominates

    def test_safety_dominance_partial_completion(
        self,
        calculator: RewardCalculator,
    ) -> None:
        """Test safety dominance with partial completion."""
        # State with no completed phases
        state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="test",
            step_count=0,
            task_id="arrival",
            completed_phases=[],
            metadata={"safety_violation": True},
        )
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27L",
        )
        obs = Observation(
            result="runway_incursion",
            score=0.0,
            phraseology_ok=False,
            issues=["runway_incursion"],
        )
        signal, total = calculator.compute_reward(state, action, obs)
        assert total == 0.0

    def test_completion_fraction(
        self,
        calculator: RewardCalculator,
        safe_action: Action,
        safe_obs: Observation,
    ) -> None:
        """Test completion scoring as fraction of phases completed."""
        # State with 5 completed phases out of 11
        state = LifecycleState(
            phase=LifecyclePhase.TAXI_IN,
            aircraft_states={},
            episode_id="test",
            step_count=10,
            task_id="arrival",
            completed_phases=[
                LifecyclePhase.APPROACH,
                LifecyclePhase.LANDING,
                LifecyclePhase.ARRIVAL_HANDOFF,
                LifecyclePhase.TAXI_IN,
                LifecyclePhase.DOCKING,
            ],
            metadata={},
        )
        signal, total = calculator.compute_reward(state, safe_action, safe_obs)
        # 5 completed / 11 total = ~0.45
        assert signal.completion > 0.0

    def test_efficiency_inverse_steps(
        self,
        calculator: RewardCalculator,
        safe_action: Action,
        safe_obs: Observation,
    ) -> None:
        """Test efficiency: fewer steps = higher score."""
        # Optimal steps = 11 (all phases)
        optimal_state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="test",
            step_count=11,  # Optimal
            task_id="arrival",
            completed_phases=[],
            metadata={},
        )
        # Many extra steps
        slow_state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="test",
            step_count=50,  # Inefficient
            task_id="arrival",
            completed_phases=[],
            metadata={},
        )
        _, optimal_total = calculator.compute_reward(
            optimal_state, safe_action, safe_obs
        )
        _, slow_total = calculator.compute_reward(slow_state, safe_action, safe_obs)
        assert optimal_total > slow_total

    def test_communication_from_phraseology_ok(
        self,
        calculator: RewardCalculator,
        safe_state: LifecycleState,
        safe_action: Action,
    ) -> None:
        """Test communication score comes from phraseology_ok."""
        good_obs = Observation(
            result="clear",
            score=1.0,
            phraseology_ok=True,
            issues=[],
        )
        bad_obs = Observation(
            result="clear",
            score=1.0,
            phraseology_ok=False,
            issues=["phraseology_error"],
        )
        good_signal, _ = calculator.compute_reward(safe_state, safe_action, good_obs)
        bad_signal, _ = calculator.compute_reward(safe_state, safe_action, bad_obs)
        assert good_signal.communication > bad_signal.communication

    def test_weights_sum_to_one(self) -> None:
        """Test that weights sum to 1.0."""
        calc = RewardCalculator()
        total_weight = calc.weights["safety"]
        total_weight += calc.weights["legality"]
        total_weight += calc.weights["completion"]
        total_weight += calc.weights["efficiency"]
        total_weight += calc.weights["communication"]
        assert abs(total_weight - 1.0) < 1e-9

    def test_legality_violation_penalty(self) -> None:
        """Test that protocol violations reduce legality score."""
        calc = RewardCalculator()
        state = LifecycleState(
            phase=LifecyclePhase.LANDING,
            aircraft_states={},
            episode_id="test",
            step_count=0,
            task_id="arrival",
            completed_phases=[],
            metadata={},
        )
        action = Action(
            clearance_type=ClearanceType.TAKEOFF,  # Wrong clearance for landing
            target_callsign="BAW123",
        )
        obs = Observation(
            result="protocol_violation",
            score=0.0,
            phraseology_ok=False,
            issues=["invalid_clearance"],
        )
        signal, total = calc.compute_reward(state, action, obs)
        assert signal.legality == 0.0


class TestTaskGrader:
    """Tests for TaskGrader."""

    @pytest.fixture
    def grader(self) -> TaskGrader:
        """Create a TaskGrader instance."""
        return TaskGrader()

    def test_grade_episode_returns_float(self, grader: TaskGrader) -> None:
        """Test that grade_episode returns a float."""
        result = grader.grade_episode([0.5, 0.6, 0.7], "arrival")
        assert isinstance(result, float)

    def test_grade_episode_in_range(self, grader: TaskGrader) -> None:
        """Test that grade is in [0.0, 1.0]."""
        rewards = [0.5, 0.6, 0.7, 0.8]
        result = grader.grade_episode(rewards, "arrival")
        assert 0.0 <= result <= 1.0

    def test_grade_episode_all_zeros(self, grader: TaskGrader) -> None:
        """Test grading episode with all zero rewards."""
        result = grader.grade_episode([0.0, 0.0, 0.0], "arrival")
        assert result == 0.0

    def test_grade_episode_all_ones(self, grader: TaskGrader) -> None:
        """Test grading episode with all perfect rewards."""
        result = grader.grade_episode([1.0, 1.0, 1.0], "arrival")
        assert result == 1.0

    def test_grade_episode_averages(self, grader: TaskGrader) -> None:
        """Test that grading averages the episode rewards."""
        rewards = [0.4, 0.6, 0.8]
        result = grader.grade_episode(rewards, "arrival")
        expected = (0.4 + 0.6 + 0.8) / 3
        assert abs(result - expected) < 1e-9

    def test_grade_episode_with_safety_failure(self, grader: TaskGrader) -> None:
        """Test that safety failures propagate through grading."""
        # Even if later rewards are high, early safety failure dominates
        rewards = [0.0, 0.8, 0.9, 1.0]
        result = grader.grade_episode(rewards, "arrival")
        # First reward is 0 due to safety dominance
        assert result < 0.9

    def test_grade_episode_empty_rewards(self, grader: TaskGrader) -> None:
        """Test grading empty episode returns 0."""
        result = grader.grade_episode([], "arrival")
        assert result == 0.0

    def test_deterministic_grading(self, grader: TaskGrader) -> None:
        """Test that grading is deterministic (pure function)."""
        rewards = [0.5, 0.6, 0.7, 0.8]
        result1 = grader.grade_episode(rewards, "arrival")
        result2 = grader.grade_episode(rewards, "arrival")
        assert result1 == result2

    def test_grade_episode_single_reward(self, grader: TaskGrader) -> None:
        """Test grading episode with single reward."""
        result = grader.grade_episode([0.75], "departure")
        assert result == 0.75


class TestIntegration:
    """Integration tests for reward system."""

    def test_safe_episode_full_completion(self) -> None:
        """Test a full safe episode gets high score."""
        calc = RewardCalculator()
        grader = TaskGrader()

        # Simulate a safe episode with high scores
        episode_rewards = []
        state = LifecycleState(
            phase=LifecyclePhase.APPROACH,
            aircraft_states={},
            episode_id="test",
            step_count=0,
            task_id="arrival",
            completed_phases=[],
            metadata={},
        )
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27L",
        )

        # Multiple steps with safe observations
        for _ in range(5):
            obs = Observation(
                result="clear",
                score=1.0,
                phraseology_ok=True,
                issues=[],
            )
            signal, total = calc.compute_reward(state, action, obs)
            episode_rewards.append(total)

        final_score = grader.grade_episode(episode_rewards, "arrival")
        assert 0.0 <= final_score <= 1.0

    def test_unsafe_episode_low_score(self) -> None:
        """Test an unsafe episode gets low score."""
        calc = RewardCalculator()
        grader = TaskGrader()

        episode_rewards = []
        state = LifecycleState(
            phase=LifecyclePhase.LANDING,
            aircraft_states={},
            episode_id="test",
            step_count=0,
            task_id="arrival",
            completed_phases=[LifecyclePhase.APPROACH],
            metadata={"safety_violation": True},
        )
        action = Action(
            clearance_type=ClearanceType.LANDING,
            target_callsign="BAW123",
            runway="27L",
        )
        obs = Observation(
            result="collision",
            score=0.0,
            phraseology_ok=False,
            issues=["collision"],
        )
        signal, total = calc.compute_reward(state, action, obs)
        episode_rewards.append(total)

        final_score = grader.grade_episode(episode_rewards, "arrival")
        assert final_score == 0.0
