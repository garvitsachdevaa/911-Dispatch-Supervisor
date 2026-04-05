"""Read-only 2D visualizer synchronized to environment state."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pydantic import BaseModel, Field, PrivateAttr

from src.models import (
    IncidentSeverity,
    IncidentStatus,
    State,
    UnitStatus,
    UnitType,
)


class Viewer2D(BaseModel):
    """Read-only 2D visualizer synchronized to environment state.

    Uses Matplotlib with non-interactive Agg backend. No event handlers
    are registered — purely observational viewer.
    """

    model_config = {"extra": "forbid"}

    grid_line_alpha: float = Field(default=0.25, ge=0.0, le=1.0)
    figure_width_inches: float = Field(default=12.0, gt=0.0)
    figure_height_inches: float = Field(default=10.0, gt=0.0)

    units: dict = Field(default_factory=dict)
    incidents: dict = Field(default_factory=dict)
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""
    city_time: float = 0.0
    grid_size: tuple[int, int] = (100, 100)

    _figure: plt.Figure = PrivateAttr()
    _axes: plt.Axes = PrivateAttr()
    _canvas: FigureCanvasAgg = PrivateAttr()

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._setup_figure()

    def _setup_figure(self) -> None:
        self._figure = plt.figure(
            figsize=(self.figure_width_inches, self.figure_height_inches)
        )
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvasAgg(self._figure)

    @property
    def figure(self) -> plt.Figure:
        return self._figure

    @property
    def axes(self) -> plt.Axes:
        return self._axes

    @property
    def canvas(self) -> FigureCanvasAgg:
        return self._canvas

    def update(self, state: State) -> None:
        self.units = dict(state.units)
        self.incidents = dict(state.incidents)
        self.episode_id = state.episode_id
        self.step_count = state.step_count
        self.task_id = state.task_id
        self.city_time = float(state.city_time)

        grid = state.metadata.get("grid_size")
        if isinstance(grid, (list, tuple)) and len(grid) >= 2:
            self.grid_size = (int(grid[0]), int(grid[1]))

    def render(self) -> bytes:
        self._clear_axes()
        self._draw_city_grid()
        self._draw_incidents()
        self._draw_units()
        self._draw_header()
        self._apply_styling()
        self._canvas.draw()
        buf = BytesIO()
        self._figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf.read()

    def render_to_file(self, path: str, state: State) -> None:
        self.update(state)
        self._clear_axes()
        self._draw_city_grid()
        self._draw_incidents()
        self._draw_units()
        self._draw_header()
        self._apply_styling()
        self._figure.savefig(path, format="png", bbox_inches="tight")

    def _clear_axes(self) -> None:
        self._axes.clear()

    def _draw_city_grid(self) -> None:
        width, height = self.grid_size
        self._axes.set_xlim(-1, width + 1)
        self._axes.set_ylim(-1, height + 1)

        # Light grid
        for x in range(0, width + 1, max(1, width // 10)):
            self._axes.plot([x, x], [0, height], color="#999999", alpha=self.grid_line_alpha, linewidth=0.8)
        for y in range(0, height + 1, max(1, height // 10)):
            self._axes.plot([0, width], [y, y], color="#999999", alpha=self.grid_line_alpha, linewidth=0.8)

    def _draw_units(self) -> None:
        marker_by_type: dict[UnitType, str] = {
            UnitType.MEDIC: "s",
            UnitType.ENGINE: "^",
            UnitType.LADDER: "^",
            UnitType.PATROL: "o",
            UnitType.HAZMAT: "D",
            UnitType.SUPERVISOR: "P",
        }
        color_by_status: dict[UnitStatus, str] = {
            UnitStatus.AVAILABLE: "#2E7D32",
            UnitStatus.DISPATCHED: "#C62828",
            UnitStatus.ON_SCENE: "#1565C0",
            UnitStatus.TRANSPORTING: "#6A1B9A",
            UnitStatus.OUT_OF_SERVICE: "#616161",
        }

        for unit in self.units.values():
            marker = marker_by_type.get(unit.unit_type, "o")
            color = color_by_status.get(unit.status, "#000000")
            self._axes.scatter(
                [unit.location_x],
                [unit.location_y],
                marker=marker,
                s=90,
                c=color,
                edgecolors="#111111",
                linewidths=0.6,
                zorder=20,
            )
            self._axes.text(
                unit.location_x + 0.8,
                unit.location_y + 0.8,
                unit.unit_id,
                fontsize=7,
                zorder=21,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.75),
            )

            if unit.assigned_incident_id and unit.assigned_incident_id in self.incidents:
                inc = self.incidents[unit.assigned_incident_id]
                self._axes.plot(
                    [unit.location_x, inc.location_x],
                    [unit.location_y, inc.location_y],
                    color="#333333",
                    alpha=0.35,
                    linewidth=1.0,
                    zorder=10,
                )

    def _draw_incidents(self) -> None:
        color_by_sev: dict[IncidentSeverity, str] = {
            IncidentSeverity.PRIORITY_1: "#D32F2F",
            IncidentSeverity.PRIORITY_2: "#F57C00",
            IncidentSeverity.PRIORITY_3: "#FBC02D",
        }

        for inc in self.incidents.values():
            if inc.status == IncidentStatus.RESOLVED:
                continue
            color = color_by_sev.get(inc.severity, "#000000")
            self._axes.scatter(
                [inc.location_x],
                [inc.location_y],
                marker="X",
                s=120,
                c=color,
                edgecolors="#111111",
                linewidths=0.8,
                zorder=25,
            )
            label = f"{inc.incident_id} {inc.incident_type.value} {inc.severity.value}"
            self._axes.text(
                inc.location_x + 0.8,
                inc.location_y - 1.2,
                label,
                fontsize=7,
                zorder=26,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.8),
            )

    def _draw_header(self) -> None:
        info = (
            f"E: {self.episode_id} | step: {self.step_count} | t: {self.city_time:.0f}s | task: {self.task_id}"
        )
        self._axes.text(
            0.02,
            0.98,
            info,
            transform=self._axes.transAxes,
            fontsize=9,
            va="top",
            ha="left",
            color="#111111",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", alpha=0.85),
            zorder=30,
        )

    def _apply_styling(self) -> None:
        self._axes.set_aspect("equal")
        self._axes.set_facecolor("#F7F7F7")
        self._figure.patch.set_facecolor("#FFFFFF")
        self._axes.set_xlabel("X (blocks)", fontsize=9)
        self._axes.set_ylabel("Y (blocks)", fontsize=9)
        self._axes.tick_params(labelsize=7)
        self._axes.grid(True, alpha=0.3, linestyle=":", color="#999999")
        self._axes.set_title(
            "911 Dispatch Supervisor — 2D Visualizer",
            fontsize=12,
            fontweight="bold",
            pad=10,
        )
