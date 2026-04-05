"""Task implementations for the 911 dispatch supervisor environment."""

# NOTE: Avoid importing task modules here.
# The state machine imports `src.tasks.registry`, and importing this package-level
# module must not trigger imports that depend on the state machine (circular).

__all__ = ["SingleIncidentTask", "MultiIncidentTask", "MassCasualtyTask", "ShiftSurgeTask"]
