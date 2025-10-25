"""Compatibility wrapper for generating layered structures via the package API."""
from __future__ import annotations

from pathlib import Path

from layer_toolkit.config import load_settings
from layer_toolkit.generation.layers import LayerGenerationRequest, LayerGenerator


def main() -> None:
    settings = load_settings()
    element = input("Which element? ").strip()
    structure_type = input("BCC or HCP? ").strip()
    layers_input = input("Which layers? (space separated) ")
    layers = [int(value) for value in layers_input.split() if value.isdigit()]

    if not layers:
        raise ValueError("At least one valid layer count is required")

    request = LayerGenerationRequest(
        element=element,
        structure_type=structure_type,
        layer_counts=layers,
        submit_jobs=True,
    )

    generator = LayerGenerator(settings, base_directory=Path.cwd())
    generator.run(request)


if __name__ == "__main__":
    main()
