"""Tests for repository structure and package layout."""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_src_directory_exists():
    assert (ROOT / "src").is_dir()


def test_all_src_modules_exist():
    expected = [
        "src/__init__.py",
        "src/models.py",
        "src/protocol.py",
        "src/physics.py",
        "src/airport_schema.py",
        "src/state_machine.py",
        "src/api.py",
        "src/rewards.py",
        "src/phraseology.py",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_tasks_subpackage_exists():
    assert (ROOT / "src/tasks").is_dir()
    expected = [
        "src/tasks/__init__.py",
        "src/tasks/registry.py",
        "src/tasks/arrival.py",
        "src/tasks/departure.py",
        "src/tasks/integrated.py",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_server_subpackage_exists():
    assert (ROOT / "src/server").is_dir()
    expected = [
        "src/server/__init__.py",
        "src/server/app.py",
        "src/server/requirements.txt",
        "src/server/Dockerfile",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_visualizer_subpackage_exists():
    assert (ROOT / "src/visualizer").is_dir()
    expected = [
        "src/visualizer/__init__.py",
        "src/visualizer/viewer.py",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_tests_directory_exists():
    assert (ROOT / "tests").is_dir()
    assert (ROOT / "tests/__init__.py").is_file()


def test_pyproject_toml_exists():
    assert (ROOT / "pyproject.toml").is_file()


def test_gatwick_ground_control_removed():
    assert not (ROOT / "gatwick_ground_control").exists(), (
        "Old package should be removed"
    )


def test_src_package_importable():
    import src

    assert hasattr(src, "__version__")


def test_all_src_modules_importable():
    modules = [
        "src.models",
        "src.protocol",
        "src.physics",
        "src.airport_schema",
        "src.state_machine",
        "src.api",
        "src.rewards",
        "src.phraseology",
    ]
    for name in modules:
        import importlib

        m = importlib.import_module(name)
        assert m is not None


def test_tasks_subpackage_importable():
    modules = [
        "src.tasks",
        "src.tasks.registry",
        "src.tasks.arrival",
        "src.tasks.departure",
        "src.tasks.integrated",
    ]
    for name in modules:
        import importlib

        m = importlib.import_module(name)
        assert m is not None


def test_visualizer_subpackage_importable():
    import importlib

    m = importlib.import_module("src.visualizer")
    assert m is not None
    v = importlib.import_module("src.visualizer.viewer")
    assert v is not None


def test_openenv_yaml_exists():
    assert (ROOT / "openenv.yaml").is_file()
