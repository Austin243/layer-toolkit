"""Compatibility wrapper for ELFCAR directory analysis."""
from __future__ import annotations

from pathlib import Path

from layer_toolkit.analysis.elf import analyze_directory


def main() -> None:
    directory = Path.cwd()
    results = analyze_directory(directory)
    if not results:
        print("No ELFCAR files found in current directory")
        return

    data_path = directory / "elfcar_data.dat"
    coords_path = directory / "elfcar_coords.dat"

    data_lines = ["Layers\tMaxELF\tDist\tAvgELF\n"]
    coord_lines = ["Layers\tMaxFracCoord\tMaxCartCoord\n"]

    for result in results:
        metrics = result.metrics
        data_lines.append(
            f"{result.label}\t{metrics.max_elf:.5f}\t{metrics.shortest_distance:.5f}\t{metrics.average_elf:.5f}\n"
        )
        frac = "\t".join(f"{value:.5f}" for value in metrics.max_frac_coord)
        cart = "\t".join(f"{value:.5f}" for value in metrics.max_cart_coord)
        coord_lines.append(f"{result.label}\t{frac}\t{cart}\n")

    data_path.write_text("".join(data_lines), encoding="utf-8")
    coords_path.write_text("".join(coord_lines), encoding="utf-8")
    print(f"Data written to {data_path} and {coords_path}")


if __name__ == "__main__":
    main()
