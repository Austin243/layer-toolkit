"""Helpers for working with bundled template resources."""
from __future__ import annotations

from importlib import resources
from pathlib import Path


def read_text(path_str: str) -> str:
    """Read a template, supporting both filesystem and package resources."""

    path = Path(path_str)
    if path.exists():
        return path.read_text(encoding="utf-8")

    try:
        package = resources.files("layer_toolkit.resources")
        relative = Path(path_str).name
        return (package / relative).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:  # pragma: no cover - I/O error path
        raise FileNotFoundError(f"Template not found: {path_str}") from exc
