"""Tests for repository structure and package layout."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_src_directory_exists() -> None:
    assert (ROOT / "src").is_dir()


def test_all_src_modules_exist() -> None:
    expected = [
        "src/__init__.py",
        "src/models.py",
        "src/protocol.py",
        "src/physics.py",
        "src/city_schema.py",
        "src/state_machine.py",
        "src/api.py",
        "src/rewards.py",
        "src/phraseology.py",
        "src/openenv_environment.py",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_tasks_subpackage_exists() -> None:
    assert (ROOT / "src/tasks").is_dir()
    expected = [
        "src/tasks/__init__.py",
        "src/tasks/registry.py",
        "src/tasks/single_incident.py",
        "src/tasks/multi_incident.py",
        "src/tasks/mass_casualty.py",
        "src/tasks/shift_surge.py",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_server_subpackage_exists() -> None:
    assert (ROOT / "src/server").is_dir()
    expected = [
        "src/server/__init__.py",
        "src/server/app.py",
        "src/server/requirements.txt",
        "src/server/Dockerfile",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_visualizer_subpackage_exists() -> None:
    assert (ROOT / "src/visualizer").is_dir()
    expected = [
        "src/visualizer/__init__.py",
        "src/visualizer/viewer.py",
    ]
    for path in expected:
        assert (ROOT / path).is_file(), f"Missing: {path}"


def test_pyproject_toml_exists() -> None:
    assert (ROOT / "pyproject.toml").is_file()


def test_openenv_yaml_exists() -> None:
    assert (ROOT / "openenv.yaml").is_file()


def test_src_package_importable() -> None:
    import src

    assert hasattr(src, "__version__")


def test_all_src_modules_importable() -> None:
    modules = [
        "src.models",
        "src.protocol",
        "src.physics",
        "src.city_schema",
        "src.state_machine",
        "src.api",
        "src.rewards",
        "src.phraseology",
        "src.openenv_environment",
    ]
    for name in modules:
        import importlib

        assert importlib.import_module(name) is not None
