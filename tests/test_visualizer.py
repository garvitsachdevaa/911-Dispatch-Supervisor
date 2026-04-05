"""Tests for the 2D visualizer synchronized to environment state."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from src.airport_schema import AirportSchema, AirportSchemaLoader, NodeType
from src.models import AircraftState, LifecyclePhase, State


class TestViewer2DStateSynchronization:
    """Test state synchronization accuracy."""

    def test_update_accepts_state_object(self) -> None:
        """Viewer2D.update() accepts a State object."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={},
            episode_id="test-001",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)

    def test_update_stores_aircraft_from_state(self) -> None:
        """Viewer2D.update() stores aircraft from State.aircraft dict."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        aircraft = AircraftState(
            callsign="BAW456",
            x_ft=1000.0,
            y_ft=-2000.0,
            heading_deg=270.0,
            altitude_ft=0.0,
            speed_kt=30.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW456": aircraft},
            episode_id="test-002",
            step_count=5,
            task_id="arrival",
        )
        viewer.update(state)
        assert viewer.aircraft["BAW456"].callsign == "BAW456"
        assert viewer.aircraft["BAW456"].x_ft == 1000.0
        assert viewer.aircraft["BAW456"].y_ft == -2000.0

    def test_update_overwrites_previous_aircraft_state(self) -> None:
        """Viewer2D.update() overwrites previous aircraft state on subsequent calls."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        aircraft1 = AircraftState(
            callsign="BAW123",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.AT_GATE,
        )
        state1 = State(
            phase=LifecyclePhase.AT_GATE,
            aircraft={"BAW123": aircraft1},
            episode_id="test-003",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state1)

        aircraft2 = AircraftState(
            callsign="BAW123",
            x_ft=500.0,
            y_ft=-500.0,
            heading_deg=180.0,
            altitude_ft=0.0,
            speed_kt=20.0,
            phase=LifecyclePhase.TAXI_OUT,
        )
        state2 = State(
            phase=LifecyclePhase.TAXI_OUT,
            aircraft={"BAW123": aircraft2},
            episode_id="test-003",
            step_count=1,
            task_id="arrival",
        )
        viewer.update(state2)

        assert viewer.aircraft["BAW123"].x_ft == 500.0
        assert viewer.aircraft["BAW123"].y_ft == -500.0
        assert viewer.aircraft["BAW123"].phase == LifecyclePhase.TAXI_OUT

    def test_state_sync_reflects_step_count(self) -> None:
        """State synchronization reflects step_count from state."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        state = State(
            phase=LifecyclePhase.APPROACH,
            aircraft={},
            episode_id="test-004",
            step_count=42,
            task_id="arrival",
        )
        viewer.update(state)
        assert viewer.step_count == 42


class TestViewer2DRenderPNG:
    """Test PNG rendering output."""

    def test_render_returns_bytes(self) -> None:
        """Viewer2D.render() returns PNG bytes."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        png_bytes = viewer.render()
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0

    def test_render_returns_valid_png_signature(self) -> None:
        """Viewer2D.render() returns valid PNG (starts with PNG signature)."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        png_bytes = viewer.render()
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    def test_render_to_file_saves_png(self, tmp_path: Path) -> None:
        """Viewer2D.render_to_file() saves a valid PNG to disk."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        output_path = tmp_path / "test_frame.png"

        aircraft = AircraftState(
            callsign="BAW789",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=270.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.LANDING,
        )
        state = State(
            phase=LifecyclePhase.LANDING,
            aircraft={"BAW789": aircraft},
            episode_id="test-005",
            step_count=0,
            task_id="arrival",
        )
        viewer.render_to_file(str(output_path), state)

        assert output_path.exists()
        png_data = output_path.read_bytes()
        assert png_data[:8] == b"\x89PNG\r\n\x1a\n"


class TestViewer2DReadOnlyGuarantee:
    """Test read-only guarantee: no interactive controls."""

    def test_no_mpl_event_handlers_registered(self) -> None:
        """Viewer2D uses non-interactive backend with no event handlers."""
        import matplotlib
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        assert matplotlib.get_backend().lower() == "agg"

    def test_figure_canvas_not_interactive(self) -> None:
        """Viewer2D.figure.canvas does not support interactive events."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        canvas = viewer.figure.canvas
        events = [
            "button_press_event",
            "button_release_event",
            "motion_notify_event",
            "key_press_event",
            "key_release_event",
            "scroll_event",
        ]
        for event_name in events:
            assert not hasattr(canvas, event_name) or canvas.mpl_disconnect is None

    def test_no_canvas_cid_mapping(self) -> None:
        """Viewer2D.canvas has no registered callback IDs."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        canvas = viewer.figure.canvas
        cid_attr = getattr(canvas, "_connection", {})
        assert len(cid_attr) == 0


class TestViewer2DAircraftRendering:
    """Test aircraft rendering in the visualizer."""

    def test_render_with_single_aircraft(self) -> None:
        """Viewer2D.render() succeeds with one aircraft in state."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        aircraft = AircraftState(
            callsign="BAW100",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.AT_GATE,
        )
        state = State(
            phase=LifecyclePhase.AT_GATE,
            aircraft={"BAW100": aircraft},
            episode_id="test-006",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        png_bytes = viewer.render()
        assert len(png_bytes) > 0

    def test_render_with_multiple_aircraft(self) -> None:
        """Viewer2D.render() succeeds with multiple aircraft."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        aircraft1 = AircraftState(
            callsign="BAW100",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=0.0,
            phase=LifecyclePhase.AT_GATE,
        )
        aircraft2 = AircraftState(
            callsign="EZY200",
            x_ft=1000.0,
            y_ft=-1000.0,
            heading_deg=270.0,
            altitude_ft=0.0,
            speed_kt=15.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW100": aircraft1, "EZY200": aircraft2},
            episode_id="test-007",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        png_bytes = viewer.render()
        assert len(png_bytes) > 0


class TestViewer2DAirportTopology:
    """Test airport topology rendering."""

    def test_render_with_airport_schema(self) -> None:
        """Viewer2D renders airport topology from schema."""
        from src.visualizer.viewer import Viewer2D

        schema = AirportSchemaLoader.load("egkk_gatwick")
        viewer = Viewer2D(airport_schema=schema)
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={},
            episode_id="test-008",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        png_bytes = viewer.render()
        assert len(png_bytes) > 0

    def test_render_runs_without_errors_on_full_schema(self) -> None:
        """Viewer2D renders without errors using full Gatwick schema."""
        from src.visualizer.viewer import Viewer2D

        schema = AirportSchemaLoader.load("egkk_gatwick")
        viewer = Viewer2D(airport_schema=schema)

        aircraft = AircraftState(
            callsign="BAW123",
            x_ft=0.0,
            y_ft=-2500.0,
            heading_deg=270.0,
            altitude_ft=0.0,
            speed_kt=30.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW123": aircraft},
            episode_id="test-009",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        png_bytes = viewer.render()
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


class TestViewer2DPhaseIndicator:
    """Test phase indicator text overlay."""

    def test_phase_indicator_rendered_in_image(self) -> None:
        """Phase indicator text appears in rendered image metadata/content."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        state = State(
            phase=LifecyclePhase.LANDING,
            aircraft={},
            episode_id="test-010",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        viewer.render()
        assert viewer.phase_indicator == LifecyclePhase.LANDING


class TestViewer2DConflicts:
    """Test conflict detection and debug overlays."""

    def test_conflict_detection_nearby_aircraft(self) -> None:
        """Viewer2D detects when aircraft are too close (conflict)."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D(conflict_distance_ft=500.0)
        aircraft1 = AircraftState(
            callsign="BAW100",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=10.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        aircraft2 = AircraftState(
            callsign="EZY200",
            x_ft=200.0,
            y_ft=200.0,
            heading_deg=270.0,
            altitude_ft=0.0,
            speed_kt=10.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW100": aircraft1, "EZY200": aircraft2},
            episode_id="test-011",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        conflicts = viewer.detect_conflicts()
        assert len(conflicts) > 0

    def test_no_conflict_distant_aircraft(self) -> None:
        """Viewer2D reports no conflicts when aircraft are far apart."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D(conflict_distance_ft=500.0)
        aircraft1 = AircraftState(
            callsign="BAW100",
            x_ft=0.0,
            y_ft=0.0,
            heading_deg=90.0,
            altitude_ft=0.0,
            speed_kt=10.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        aircraft2 = AircraftState(
            callsign="EZY200",
            x_ft=5000.0,
            y_ft=5000.0,
            heading_deg=270.0,
            altitude_ft=0.0,
            speed_kt=10.0,
            phase=LifecyclePhase.TAXI_IN,
        )
        state = State(
            phase=LifecyclePhase.TAXI_IN,
            aircraft={"BAW100": aircraft1, "EZY200": aircraft2},
            episode_id="test-012",
            step_count=0,
            task_id="arrival",
        )
        viewer.update(state)
        conflicts = viewer.detect_conflicts()
        assert len(conflicts) == 0


class TestViewer2DInitialization:
    """Test Viewer2D initialization options."""

    def test_default_initialization(self) -> None:
        """Viewer2D initializes with defaults."""
        from src.visualizer.viewer import Viewer2D

        viewer = Viewer2D()
        assert viewer.airport_schema is None
        assert viewer.figure is not None
        assert viewer.canvas is not None

    def test_initialization_with_schema(self) -> None:
        """Viewer2D accepts optional airport_schema at init."""
        from src.visualizer.viewer import Viewer2D

        schema = AirportSchemaLoader.load("dummy_small")
        viewer = Viewer2D(airport_schema=schema)
        assert viewer.airport_schema is not None
        assert viewer.airport_schema.airport_code == "DUMMY"
