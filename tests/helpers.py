"""Test utility helpers for model/protocol/physics/schema tests."""

from __future__ import annotations

import sys
from io import StringIO
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError

F = TypeVar("F", bound=Callable[..., Any])


def capture_stdout(func: F, *args: Any, **kwargs: Any) -> str:
    """Run func and capture stdout."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        func(*args, **kwargs)
        return sys.stdout.getvalue()
    finally:
        sys.stdout = old_stdout


def assert_valid_model(
    model_class: type[BaseModel], instance_dict: dict[str, Any]
) -> None:
    """Assert model accepts valid data without raising ValidationError."""
    model_class(**instance_dict)


def assert_invalid_model(
    model_class: type[BaseModel], bad_instance_dict: dict[str, Any], field_name: str
) -> None:
    """Assert ValidationError is raised on specific field."""
    try:
        model_class(**bad_instance_dict)
    except ValidationError as e:
        assert field_name in str(e), f"Expected error on field '{field_name}', got: {e}"
        return
    raise AssertionError(
        f"Expected ValidationError for field '{field_name}', but none was raised"
    )
