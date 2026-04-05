"""Read-only 2D visualizer synchronized to environment state."""

from io import BytesIO
from math import cos, radians, sin
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from pydantic import BaseModel, Field, PrivateAttr

from src.airport_schema import AirportSchema, NodeType
from src.models import AircraftState, LifecyclePhase, State


class Viewer2D(BaseModel):
    """Read-only 2D visualizer synchronized to environment state.

    Uses Matplotlib with non-interactive Agg backend. No event handlers
    are registered — purely observational viewer.
    """

    model_config = {"extra": "forbid"}

    airport_schema: AirportSchema | None = None
    conflict_distance_ft: float = Field(default=300.0, ge=0.0)
    figure_width_inches: float = Field(default=12.0, gt=0.0)
    figure_height_inches: float = Field(default=10.0, gt=0.0)

    aircraft: dict[str, AircraftState] = Field(default_factory=dict)
    phase_indicator: LifecyclePhase | None = None
    episode_id: str = ""
    step_count: int = 0
    task_id: str = ""

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
        self.aircraft = dict(state.aircraft)
        self.phase_indicator = state.phase
        self.episode_id = state.episode_id
        self.step_count = state.step_count
        self.task_id = state.task_id

    def render(self) -> bytes:
        self._clear_axes()
        self._draw_airport_topology()
        self._draw_aircraft()
        self._draw_phase_indicator()
        self._apply_styling()
        self._canvas.draw()
        buf = BytesIO()
        self._figure.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        return buf.read()

    def render_to_file(self, path: str, state: State) -> None:
        self.update(state)
        self._clear_axes()
        self._draw_airport_topology()
        self._draw_aircraft()
        self._draw_phase_indicator()
        self._apply_styling()
        self._figure.savefig(path, format="png", bbox_inches="tight")

    def detect_conflicts(self) -> list[tuple[str, str]]:
        conflicts: list[tuple[str, str]] = []
        aircraft_list = list(self.aircraft.values())
        for i, ac1 in enumerate(aircraft_list):
            for ac2 in aircraft_list[i + 1 :]:
                dx = ac1.x_ft - ac2.x_ft
                dy = ac1.y_ft - ac2.y_ft
                distance = (dx * dx + dy * dy) ** 0.5
                if distance < self.conflict_distance_ft and distance > 0:
                    conflicts.append((ac1.callsign, ac2.callsign))
        return conflicts

    def _clear_axes(self) -> None:
        self._axes.clear()

    def _draw_airport_topology(self) -> None:
        if self.airport_schema is None:
            return
        schema = self.airport_schema

        hold_short_nodes: list[tuple[float, float]] = []
        taxiway_edges: list[tuple[float, float, float, float]] = []
        gate_nodes: list[tuple[float, float, str]] = []
        stands: list[tuple[float, float, str]] = []

        for node_id, node in schema.nodes.items():
            if node.node_type == NodeType.HOLD_SHORT:
                hold_short_nodes.append((node.x_ft, node.y_ft))
            elif node.node_type == NodeType.GATE:
                gate_nodes.append((node.x_ft, node.y_ft, node.name or node_id))
            elif node.node_type == NodeType.STAND:
                stands.append((node.x_ft, node.y_ft, node.name or node_id))

        node_coords: dict[str, tuple[float, float]] = {
            node_id: (node.x_ft, node.y_ft) for node_id, node in schema.nodes.items()
        }
        for edge in schema.edges:
            if edge.movement_type.value in (
                "taxi",
                "pushback",
                "queue_join",
                "runway_transition",
                "exit_runway",
                "takeoff_run",
            ):
                if edge.from_node in node_coords and edge.to_node in node_coords:
                    x1, y1 = node_coords[edge.from_node]
                    x2, y2 = node_coords[edge.to_node]
                    taxiway_edges.append((x1, y1, x2, y2))

        for runway in schema.runways:
            threshold_x = runway.get("threshold_x", 0.0)
            threshold_y = runway.get("threshold_y", 0.0)
            length_ft = runway.get("length_ft", 3000.0)
            heading_deg = runway.get("heading_deg", 0.0)

            end_x = threshold_x + length_ft * sin(radians(heading_deg))
            end_y = threshold_y + length_ft * cos(radians(heading_deg))
            self._axes.plot(
                [threshold_x, end_x],
                [threshold_y, end_y],
                color="#222222",
                linewidth=8,
                solid_capstyle="butt",
                zorder=1,
            )

        for x1, y1, x2, y2 in taxiway_edges:
            self._axes.plot(
                [x1, x2],
                [y1, y2],
                color="#888888",
                linewidth=1.5,
                zorder=2,
            )

        for x, y in hold_short_nodes:
            self._axes.plot(
                [x - 100, x + 100],
                [y, y],
                color="#FF6600",
                linewidth=2,
                linestyle="--",
                zorder=3,
            )

        for x, y, name in gate_nodes:
            rect = plt.Rectangle(
                (x - 150, y - 100),
                300,
                200,
                fill=True,
                facecolor="#DDDDEE",
                edgecolor="#666699",
                linewidth=2,
                zorder=4,
            )
            self._axes.add_patch(rect)
            self._axes.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=7,
                color="#333366",
                zorder=5,
            )

        for x, y, name in stands:
            rect = plt.Rectangle(
                (x - 100, y - 75),
                200,
                150,
                fill=True,
                facecolor="#EEDDDD",
                edgecolor="#996666",
                linewidth=1.5,
                zorder=4,
            )
            self._axes.add_patch(rect)
            self._axes.text(
                x,
                y,
                name,
                ha="center",
                va="center",
                fontsize=7,
                color="#663333",
                zorder=5,
            )

    def _draw_aircraft(self) -> None:
        conflicts = self.detect_conflicts()
        conflict_pairs = set(conflicts)

        for callsign, ac in self.aircraft.items():
            is_conflicted = any(callsign in pair for pair in conflict_pairs)
            self._draw_single_aircraft(ac, is_conflicted)

    def _draw_single_aircraft(
        self, ac: AircraftState, conflicted: bool = False
    ) -> None:
        x = ac.x_ft
        y = ac.y_ft
        heading = radians(ac.heading_deg)
        size = 300.0

        dx = size * sin(heading)
        dy = size * cos(heading)

        triangle_x = [
            x + dx,
            x - 0.5 * dx + 0.3 * dy,
            x - 0.5 * dx - 0.3 * dy,
            x + dx,
        ]
        triangle_y = [
            y + dy,
            y - 0.5 * dy - 0.3 * dx,
            y - 0.5 * dy + 0.3 * dx,
            y + dy,
        ]

        if conflicted:
            color, edge_color = "#FF0000", "#CC0000"
        elif ac.phase in (LifecyclePhase.LANDING, LifecyclePhase.APPROACH):
            color, edge_color = "#44AA44", "#228822"
        elif ac.phase in (LifecyclePhase.TAKEOFF, LifecyclePhase.DEPARTED):
            color, edge_color = "#4488FF", "#2266CC"
        else:
            color, edge_color = "#3366CC", "#224488"

        self._axes.fill(
            triangle_x,
            triangle_y,
            color=color,
            edgecolor=edge_color,
            linewidth=1.5,
            zorder=20,
        )

        self._axes.text(
            x + 0.6 * dx,
            y + 0.6 * dy,
            ac.callsign,
            fontsize=8,
            fontweight="bold",
            color="#111111",
            ha="left",
            va="center",
            zorder=21,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", alpha=0.8),
        )

        info_text = f"{ac.altitude_ft:.0f}ft {ac.speed_kt:.0f}kt {ac.heading_deg:.0f}°"
        self._axes.text(
            x + 0.5 * dx,
            y + 0.5 * dy - 200,
            info_text,
            fontsize=6,
            color="#444444",
            ha="left",
            va="top",
            zorder=21,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFEE", alpha=0.85),
        )

    def _draw_phase_indicator(self) -> None:
        if self.phase_indicator is None:
            phase_text = "NO STATE"
        else:
            phase_text = self.phase_indicator.value.upper().replace("_", " ")

        info_text = f"E: {self.episode_id} | S: {self.step_count} | T: {self.task_id}"

        self._axes.text(
            0.02,
            0.98,
            phase_text,
            transform=self._axes.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
            color="#003366",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="#CCEECC",
                alpha=0.9,
                edgecolor="#336633",
            ),
            zorder=30,
        )
        self._axes.text(
            0.02,
            0.93,
            info_text,
            transform=self._axes.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            color="#333333",
            zorder=30,
        )

    def _apply_styling(self) -> None:
        self._axes.set_aspect("equal")
        self._axes.set_facecolor("#F5F5F0")
        self._figure.patch.set_facecolor("#EEEEEE")

        x_min, x_max = self._axes.get_xlim()
        y_min, y_max = self._axes.get_ylim()
        if self.airport_schema is not None:
            all_x = [n.x_ft for n in self.airport_schema.nodes.values()]
            all_y = [n.y_ft for n in self.airport_schema.nodes.values()]
            if all_x and all_y:
                x_min = min(all_x) - 1000
                x_max = max(all_x) + 1000
                y_min = min(all_y) - 1000
                y_max = max(all_y) + 1000
        if not self.aircraft:
            x_min = max(x_min, -10000)
            x_max = min(x_max, 10000)
            y_min = max(y_min, -10000)
            y_max = min(y_max, 10000)
        else:
            ac_x = [ac.x_ft for ac in self.aircraft.values()]
            ac_y = [ac.y_ft for ac in self.aircraft.values()]
            x_min = min(min(ac_x) - 2000, x_min)
            x_max = max(max(ac_x) + 2000, x_max)
            y_min = min(min(ac_y) - 2000, y_min)
            y_max = max(max(ac_y) + 2000, y_max)

        self._axes.set_xlim(x_min, x_max)
        self._axes.set_ylim(y_min, y_max)
        self._axes.set_xlabel("X (feet)", fontsize=9)
        self._axes.set_ylabel("Y (feet)", fontsize=9)
        self._axes.tick_params(labelsize=7)
        self._axes.grid(True, alpha=0.3, linestyle=":", color="#999999")
        self._axes.set_title(
            "ATC Ground Control — 2D Visualizer", fontsize=12, fontweight="bold", pad=10
        )
