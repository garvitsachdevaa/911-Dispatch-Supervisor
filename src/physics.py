"""City-grid physics utilities for the 911 dispatch supervisor environment."""

from __future__ import annotations

from src.models import IncidentState, UnitState, UnitStatus


CITY_BLOCK_FT = 264.0


def compute_eta(unit: UnitState, incident: IncidentState, unit_speed: float) -> float:
    """Compute Manhattan-distance ETA in seconds.

    Args:
        unit: Current unit state.
        incident: Target incident state.
        unit_speed: Speed in blocks / second.
    """
    speed = max(float(unit_speed), 1e-6)
    dx = abs(float(unit.location_x) - float(incident.location_x))
    dy = abs(float(unit.location_y) - float(incident.location_y))
    distance_blocks = dx + dy
    return distance_blocks / speed


def move_unit_toward(
    unit: UnitState,
    incident: IncidentState,
    unit_speed: float,
    dt: float,
) -> UnitState:
    """Advance unit position toward incident by dt seconds.

    Uses Manhattan (grid) movement: consume movement along x, then y.
    """
    speed = max(float(unit_speed), 0.0)
    remaining = max(0.0, speed * float(dt))

    ux = float(unit.location_x)
    uy = float(unit.location_y)
    ix = float(incident.location_x)
    iy = float(incident.location_y)

    dx = ix - ux
    step_x = max(-remaining, min(remaining, dx))
    ux += step_x
    remaining -= abs(step_x)

    if remaining > 0.0:
        dy = iy - uy
        step_y = max(-remaining, min(remaining, dy))
        uy += step_y

    return unit.model_copy(update={"location_x": ux, "location_y": uy})


def check_arrival(
    unit: UnitState,
    incident: IncidentState,
    threshold_blocks: float = 0.5,
) -> bool:
    """Return True if unit is considered arrived on scene."""
    dx = abs(float(unit.location_x) - float(incident.location_x))
    dy = abs(float(unit.location_y) - float(incident.location_y))
    return (dx + dy) <= float(threshold_blocks)


def compute_coverage_score(
    units: dict[str, UnitState],
    grid_size: tuple[int, int],
    bins_x: int = 4,
) -> float:
    """Score geographic distribution of units across the x-axis.

    This is intentionally simple and deterministic: slice the city width into bins and
    compute fraction of bins that contain at least one AVAILABLE unit.
    """
    width = max(1, int(grid_size[0]))
    bins = max(1, int(bins_x))
    bin_width = width / bins

    covered: set[int] = set()
    for unit in units.values():
        if unit.status != UnitStatus.AVAILABLE:
            continue
        idx = int(min(bins - 1, max(0.0, float(unit.location_x)) // max(bin_width, 1e-6)))
        covered.add(idx)

    return len(covered) / bins
