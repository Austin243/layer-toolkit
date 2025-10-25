"""Compatibility wrapper for bond analysis."""
from __future__ import annotations

from pathlib import Path

from layer_toolkit.analysis.bonds import analyze_poscar


def main() -> None:
    output_filename = "results.dat"
    directory = Path.cwd()

    lines: list[str] = []
    for poscar_file in sorted(directory.glob("*.vasp")):
        result = analyze_poscar(poscar_file)
        lines.extend(_format_result(poscar_file, result))
        lines.append("\n" + "-" * 40 + "\n\n")

    Path(output_filename).write_text("".join(lines), encoding="utf-8")
    print(f"Bond analysis written to {output_filename}")


def _format_result(path: Path, result) -> list[str]:
    lines = [f"File: {path.name}\n", f"Number of Layers: {result.num_layers}\n\n"]

    def _section(title: str, entries) -> None:
        lines.append(f"{title}:\n")
        for entry in entries:
            lines.append(
                f"{entry.bond_type}: {entry.length:.3f} Angstrom, Count: {entry.count}\n"
            )
        lines.append("\n")

    _section("Unique in-plane bonds (unit cell)", result.unit_cell_in_plane)
    _section("Unique interlayer bonds (unit cell)", result.unit_cell_interlayer)
    _section("Unique in-plane bonds (primitive cell)", result.primitive_in_plane)
    _section("Unique interlayer bonds (primitive cell)", result.primitive_interlayer)
    _section("Unique in-plane bonds (supercell)", result.supercell_in_plane)
    _section("Unique interlayer bonds (supercell)", result.supercell_interlayer)

    return lines


if __name__ == "__main__":
    main()
