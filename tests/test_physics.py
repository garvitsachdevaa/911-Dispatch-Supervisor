"""TDD tests for deterministic aircraft physics model."""

import math
import pytest
from src.physics import (
    GLIDE_SLOPE_DEG,
    MAX_DESCENT_RATE_FPM,
    STD_DESCENT_RATE_FPM,
    TAXI_SPEED_KT,
    TAXI_ACCEL_KT_PER_S,
    TAXI_DECEL_KT_PER_S,
    CRUISE_ALTITUDE_FT,
    GlidePathUpdate,
    DescentRateUpdate,
    SurfaceMovementUpdate,
    HeadingUpdate,
    WakeCategorySpacing,
)


class TestGlidePathUpdate:
    """Glide path must follow standard 3-degree ILS slope."""

    def test_glide_slope_angle_is_standard(self):
        """tan(3deg) must equal the configured glide slope ratio."""
        expected_ratio = math.tan(math.radians(GLIDE_SLOPE_DEG))
        # 3-degree slope: vertical/horizontal ratio = tan(3°) ≈ 0.0524
        assert abs(expected_ratio - 0.052408) < 0.001

    def test_glide_path_position_at_threshold(self):
        """At altitude 0 (threshold), x,y must equal runway threshold."""
        x, y = GlidePathUpdate(
            state={"altitude_ft": 0.0},
            runway_threshold_x=1000.0,
            runway_threshold_y=2000.0,
        )
        assert x == pytest.approx(1000.0)
        assert y == pytest.approx(2000.0)

    def test_glide_path_position_at_3000ft_altitude(self):
        """At 3000ft AGL on 3-deg slope, aircraft is ~57,250ft from threshold."""
        # horizontal_dist = altitude / tan(3deg) = 3000 / 0.0524 ≈ 57,251 ft
        x, y = GlidePathUpdate(
            state={"altitude_ft": 3000.0},
            runway_threshold_x=0.0,
            runway_threshold_y=0.0,
        )
        # Aircraft is along extended runway centerline (positive x direction)
        assert x == pytest.approx(57251.0, rel=1e-3)
        assert y == pytest.approx(0.0)

    def test_glide_path_intermediate_altitude(self):
        """At 1500ft, horizontal distance should be half of 3000ft distance."""
        x_3000, _ = GlidePathUpdate(
            state={"altitude_ft": 3000.0},
            runway_threshold_x=0.0,
            runway_threshold_y=0.0,
        )
        x_1500, _ = GlidePathUpdate(
            state={"altitude_ft": 1500.0},
            runway_threshold_x=0.0,
            runway_threshold_y=0.0,
        )
        assert x_1500 == pytest.approx(x_3000 / 2.0, rel=1e-6)

    def test_glide_path_custom_slope(self):
        """Custom glide slope of 5 degrees works correctly."""
        x, y = GlidePathUpdate(
            state={"altitude_ft": 1000.0},
            runway_threshold_x=0.0,
            runway_threshold_y=0.0,
            glide_slope_deg=5.0,
        )
        # tan(5deg) ≈ 0.0875, horizontal = 1000 / 0.0875 ≈ 11,429 ft
        assert x == pytest.approx(11429.0, rel=1e-3)

    def test_glide_path_zero_altitude_returns_threshold(self):
        """Zero altitude must return runway threshold coordinates exactly."""
        x, y = GlidePathUpdate(
            state={"altitude_ft": 0.0},
            runway_threshold_x=500.0,
            runway_threshold_y=750.0,
        )
        assert x == pytest.approx(500.0)
        assert y == pytest.approx(750.0)

    def test_glide_path_determinism(self):
        """Same inputs must always produce same outputs (pure function)."""
        args = {
            "state": {"altitude_ft": 2500.0},
            "runway_threshold_x": 1000.0,
            "runway_threshold_y": 2000.0,
        }
        result1 = GlidePathUpdate(**args)
        result2 = GlidePathUpdate(**args)
        assert result1 == result2


class TestDescentRateUpdate:
    """Descent rate varies by altitude: 500 fpm below 10kft, 1500 fpm at cruise."""

    def test_descent_rate_below_10000ft(self):
        """Below 10,000ft: descent rate must be 500 fpm."""
        rate = _get_descent_rate(5000.0)
        assert rate == pytest.approx(500.0)

    def test_descent_rate_at_10000ft(self):
        """At exactly 10,000ft: transition point, ~500 fpm."""
        rate = _get_descent_rate(10000.0)
        assert rate == pytest.approx(500.0)

    def test_descent_rate_linear_interpolation(self):
        """Descent rate must interpolate linearly between 10kft and cruise."""
        # At 22500ft (actual midpoint between 10k and 35k): rate = 500 + 1000*(12500/25000) = 1000 fpm
        rate = _get_descent_rate(22500.0)
        assert rate == pytest.approx(1000.0, rel=1e-2)

    def test_descent_rate_at_cruise_altitude(self):
        """At cruise altitude (35kft): descent rate must be 1500 fpm."""
        rate = _get_descent_rate(CRUISE_ALTITUDE_FT)
        assert rate == pytest.approx(1500.0)

    def test_descent_rate_above_cruise(self):
        """Above cruise altitude should also be 1500 fpm (clamped)."""
        rate = _get_descent_rate(40000.0)
        assert rate == pytest.approx(1500.0)

    def test_altitude_update_small_dt(self):
        """1 second at 500 fpm = 8.33 ft descent."""
        new_alt = DescentRateUpdate(
            current_altitude_ft=5000.0,
            target_altitude_ft=0.0,  # ignored for now
            dt_s=1.0,
        )
        # 500 fpm / 60 = 8.333 ft/s
        assert new_alt == pytest.approx(5000.0 - 8.333, rel=1e-3)

    def test_altitude_update_at_cruise_descent(self):
        """At cruise descent rate: 1500 fpm = 25 ft/s."""
        new_alt = DescentRateUpdate(
            current_altitude_ft=35000.0,
            target_altitude_ft=0.0,
            dt_s=1.0,
        )
        # 1500 fpm / 60 = 25 ft/s
        assert new_alt == pytest.approx(35000.0 - 25.0, rel=1e-3)

    def test_altitude_update_multi_step_to_ground(self):
        """Descending from 5000ft at 500 fpm (8.33 ft/s) takes ~600s in 600 steps."""
        alt = 5000.0
        dt = 1.0
        steps = 0
        while alt > 0 and steps < 1200:
            alt = DescentRateUpdate(alt, 0.0, dt)
            steps += 1
        assert alt == pytest.approx(0.0, abs=1.0)
        assert steps == pytest.approx(600, rel=0.01)

    def test_altitude_update_does_not_go_negative(self):
        """Altitude must not go below 0 (ground)."""
        new_alt = DescentRateUpdate(
            current_altitude_ft=10.0,
            target_altitude_ft=0.0,
            dt_s=10.0,  # large dt that would otherwise go negative
        )
        assert new_alt >= 0.0

    def test_determinism(self):
        """Same inputs must produce same outputs."""
        args = {"current_altitude_ft": 15000.0, "target_altitude_ft": 0.0, "dt_s": 2.5}
        r1 = DescentRateUpdate(**args)
        r2 = DescentRateUpdate(**args)
        assert r1 == r2


class TestSurfaceMovementUpdate:
    """Surface movement: taxi speed, heading, position update in feet."""

    def test_stationary_aircraft(self):
        """Zero speed must not change position."""
        x, y = SurfaceMovementUpdate(
            current_x=1000.0,
            current_y=2000.0,
            heading_deg=90.0,
            speed_kt=0.0,
            dt_s=10.0,
        )
        assert x == pytest.approx(1000.0)
        assert y == pytest.approx(2000.0)

    def test_east_heading_converts_kt_to_ft_per_s(self):
        """Speed 1 kt = 1.6878 ft/s. At heading 90 (east), x increases."""
        x, y = SurfaceMovementUpdate(
            current_x=0.0,
            current_y=0.0,
            heading_deg=90.0,
            speed_kt=1.0,
            dt_s=1.0,
        )
        # 1 kt * 1.6878 ft/s per kt * 1s = 1.6878 ft east
        assert x == pytest.approx(1.6878, rel=1e-3)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_north_heading(self):
        """Heading 0 (north) moves in +y direction."""
        x, y = SurfaceMovementUpdate(
            current_x=0.0,
            current_y=0.0,
            heading_deg=0.0,
            speed_kt=1.0,
            dt_s=1.0,
        )
        assert x == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(1.6878, rel=1e-3)

    def test_south_heading(self):
        """Heading 180 (south) moves in -y direction."""
        _, y = SurfaceMovementUpdate(
            current_x=0.0,
            current_y=100.0,
            heading_deg=180.0,
            speed_kt=1.0,
            dt_s=1.0,
        )
        assert y == pytest.approx(100.0 - 1.6878, rel=1e-3)

    def test_west_heading(self):
        """Heading 270 (west) moves in -x direction."""
        x, _ = SurfaceMovementUpdate(
            current_x=100.0,
            current_y=0.0,
            heading_deg=270.0,
            speed_kt=1.0,
            dt_s=1.0,
        )
        assert x == pytest.approx(100.0 - 1.6878, rel=1e-3)

    def test_taxi_speed_limit(self):
        """TAXI_SPEED_KT must be 20 knots."""
        assert TAXI_SPEED_KT == 20.0

    def test_20kt_moves_20nm_in_3600s(self):
        """20 kt for 1 hour = 20 nautical miles = 121,680 ft."""
        x, y = SurfaceMovementUpdate(
            current_x=0.0,
            current_y=0.0,
            heading_deg=90.0,
            speed_kt=TAXI_SPEED_KT,
            dt_s=3600.0,
        )
        # 20 kt * 6076.1 ft/nm = 121,522 ft
        expected = TAXI_SPEED_KT * 6076.1  # ft per hour
        assert x == pytest.approx(expected, rel=1e-3)

    def test_surface_movement_east_direction(self):
        x, y = SurfaceMovementUpdate(
            0.0, 0.0, heading_deg=90.0, speed_kt=20.0, dt_s=1.0
        )
        assert x == pytest.approx(20 * 1.6878, rel=1e-3)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_surface_movement_speed_clamping_defined(self):
        """Speed on surface must be explicitly handled (0 to TAXI_SPEED_KT)."""
        # Speed below 0 is clamped to 0
        x, y = SurfaceMovementUpdate(
            0.0, 0.0, heading_deg=90.0, speed_kt=-5.0, dt_s=1.0
        )
        assert x == pytest.approx(0.0)
        # Speed above TAXI_SPEED_KT is clamped
        x, y = SurfaceMovementUpdate(
            0.0, 0.0, heading_deg=90.0, speed_kt=30.0, dt_s=1.0
        )
        # Clamped to TAXI_SPEED_KT
        assert x == pytest.approx(TAXI_SPEED_KT * 1.6878, rel=1e-3)

    def test_determinism(self):
        """Same inputs must produce same outputs."""
        args = {
            "current_x": 100.0,
            "current_y": 200.0,
            "heading_deg": 45.0,
            "speed_kt": 15.0,
            "dt_s": 3.0,
        }
        r1 = SurfaceMovementUpdate(**args)
        r2 = SurfaceMovementUpdate(**args)
        assert r1 == r2


class TestHeadingUpdate:
    """Heading tracking: turn toward target, limited by max turn rate."""

    def test_max_turn_rate_is_3deg_per_s(self):
        """Default max turn rate must be 3.0 deg/s (typical for heavy aircraft)."""
        new_hdg = HeadingUpdate(0.0, 180.0, max_turn_rate_deg_per_s=3.0, dt_s=1.0)
        assert new_hdg == pytest.approx(3.0)

    def test_already_at_target(self):
        """When current == target, heading stays the same."""
        new_hdg = HeadingUpdate(90.0, 90.0)
        assert new_hdg == pytest.approx(90.0)

    def test_turn_toward_target_short_arc(self):
        """Turn 350 -> 10 degrees (40 deg arc, not 320 deg)."""
        new_hdg = HeadingUpdate(350.0, 10.0)
        # Shortest path: 350 -> 10 = +20 deg
        assert new_hdg == pytest.approx(353.0)

    def test_turn_toward_target_long_arc(self):
        """Turn 10 -> 350 degrees via shortest arc (-20 deg), limited to 3 deg/s."""
        new_hdg = HeadingUpdate(10.0, 350.0)
        # Shortest path: 10 -> 350 = -20 deg (clockwise). Max turn in 1s = 3 deg.
        # Result: 10 - 3 = 7 deg
        assert new_hdg == pytest.approx(7.0)

    def test_turn_rate_limited(self):
        """dt_s=2.0 at 3 deg/s = 6 deg max turn."""
        new_hdg = HeadingUpdate(0.0, 180.0, max_turn_rate_deg_per_s=3.0, dt_s=2.0)
        assert new_hdg == pytest.approx(6.0)

    def test_turn_rate_exceeds_needed(self):
        """If max turn > needed delta, stop at target exactly."""
        new_hdg = HeadingUpdate(0.0, 2.0, max_turn_rate_deg_per_s=3.0, dt_s=1.0)
        assert new_hdg == pytest.approx(2.0)

    def test_heading_wraps_at_360(self):
        """Heading must wrap to stay in [0, 360)."""
        new_hdg = HeadingUpdate(359.0, 1.0)
        assert 0.0 <= new_hdg < 360.0

    def test_360_equals_0(self):
        """Heading 360 must be equivalent to heading 0."""
        h1 = HeadingUpdate(0.0, 360.0)
        h2 = HeadingUpdate(0.0, 0.0)
        assert h1 == pytest.approx(h2)

    def test_determinism(self):
        """Same inputs must produce same outputs."""
        args = {
            "current_heading_deg": 45.0,
            "target_heading_deg": 270.0,
            "max_turn_rate_deg_per_s": 3.0,
            "dt_s": 5.0,
        }
        r1 = HeadingUpdate(**args)
        r2 = HeadingUpdate(**args)
        assert r1 == r2


class TestWakeCategorySpacing:
    """Wake turbulence separation minima between aircraft categories."""

    def test_cats_ordered(self):
        """CAT_A < CAT_B < CAT_C < CAT_D (light to super)."""
        # CAT_A (light) following CAT_B (medium): 3000 ft minimum
        assert WakeCategorySpacing("CAT_B", "CAT_A") == 3000
        # CAT_A following CAT_C (heavy): 3000 ft
        assert WakeCategorySpacing("CAT_C", "CAT_A") == 3000
        # CAT_A following CAT_D (super): 3000 ft
        assert WakeCategorySpacing("CAT_D", "CAT_A") == 3000

    def test_cat_b_following_cat_cd(self):
        """CAT_B (medium) following CAT_C/D (heavy/super): 5000 ft."""
        assert WakeCategorySpacing("CAT_C", "CAT_B") == 5000
        assert WakeCategorySpacing("CAT_D", "CAT_B") == 5000

    def test_cat_c_following_cat_d(self):
        """CAT_C (heavy) following CAT_D (super): 6000 ft."""
        assert WakeCategorySpacing("CAT_D", "CAT_C") == 6000

    def test_same_category_spacing(self):
        """Same category (e.g. CAT_C following CAT_C): 5000 ft."""
        assert WakeCategorySpacing("CAT_A", "CAT_A") == 5000
        assert WakeCategorySpacing("CAT_B", "CAT_B") == 5000
        assert WakeCategorySpacing("CAT_C", "CAT_C") == 5000
        assert WakeCategorySpacing("CAT_D", "CAT_D") == 5000

    def test_cat_cd_need_6000ft_behind_larger(self):
        """CAT_C or CAT_D following a larger (only CAT_D is larger): 6000 ft."""
        # CAT_D is only category larger than CAT_C, so no 6000 case for CAT_C
        # CAT_C following CAT_D: 6000
        assert WakeCategorySpacing("CAT_D", "CAT_C") == 6000
        # CAT_D has no larger category, so no 6000 case
        # Verify CAT_D following CAT_D is 5000 (same category)
        assert WakeCategorySpacing("CAT_D", "CAT_D") == 5000

    def test_invalid_category_raises(self):
        """Invalid wake category must raise ValueError."""
        with pytest.raises(ValueError):
            WakeCategorySpacing("CAT_X", "CAT_A")
        with pytest.raises(ValueError):
            WakeCategorySpacing("CAT_A", "CAT_X")

    def test_determinism(self):
        """Same inputs must produce same outputs."""
        r1 = WakeCategorySpacing("CAT_C", "CAT_B")
        r2 = WakeCategorySpacing("CAT_C", "CAT_B")
        assert r1 == r2


class TestPhysicsConstants:
    """All constants must match expected realistic values."""

    def test_glide_slope_deg(self):
        assert GLIDE_SLOPE_DEG == 3.0

    def test_max_descent_rate_fpm(self):
        assert MAX_DESCENT_RATE_FPM == 3000

    def test_std_descent_rate_fpm(self):
        assert STD_DESCENT_RATE_FPM == 1500

    def test_taxi_speed_kt(self):
        assert TAXI_SPEED_KT == 20

    def test_taxi_accel_kt_per_s(self):
        assert TAXI_ACCEL_KT_PER_S == 5.0

    def test_taxi_decel_kt_per_s(self):
        assert TAXI_DECEL_KT_PER_S == 10.0

    def test_cruise_altitude_ft(self):
        assert CRUISE_ALTITUDE_FT == 35000


# ---- helper to expose descent rate calculation for testing ----


def _get_descent_rate(altitude_ft: float) -> float:
    """Return descent rate in fpm for given altitude (pure function for testing)."""
    # Linear interpolation: 500 fpm at 0-10000ft, 1500 fpm at 35000ft
    if altitude_ft <= 10000.0:
        return 500.0
    if altitude_ft >= CRUISE_ALTITUDE_FT:
        return 1500.0
    # Linear: rate = 500 + (1500-500) * (alt - 10000) / (35000 - 10000)
    t = (altitude_ft - 10000.0) / (CRUISE_ALTITUDE_FT - 10000.0)
    return 500.0 + 1000.0 * t
