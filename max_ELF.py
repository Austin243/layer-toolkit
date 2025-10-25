"""Compatibility wrapper for single ELFCAR analysis."""
from __future__ import annotations

from pathlib import Path

from layer_toolkit.analysis.elf import analyze_elfcar


def main() -> None:
    elfcar_path = Path.cwd() / "ELFCAR"
    if not elfcar_path.exists():
        raise FileNotFoundError(f"ELFCAR file not found at {elfcar_path}")

    metrics = analyze_elfcar(elfcar_path)
    print(f"Highest ELF Value: {metrics.max_elf:.5f}")
    print(f"Location in Fractional Coordinates: {metrics.max_frac_coord}")
    print(f"Location in Cartesian Coordinates: {metrics.max_cart_coord}")
    print(f"Shortest Distance to an Atom (Angstroms): {metrics.shortest_distance:.5f}")


if __name__ == "__main__":
    main()
