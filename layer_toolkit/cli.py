"""Command-line interface for Layer Toolkit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

from .analysis.bonds import BondAnalysisResult, analyze_poscar
from .analysis.elf import analyze_directory, analyze_elfcar
from .config import Settings, load_settings
from .generation.layers import LayerGenerationRequest, LayerGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer Toolkit CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to configuration file (defaults to config.json in CWD)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate-layers", help="Generate VASP inputs for layered structures")
    gen_parser.add_argument("--element", required=True, help="Chemical symbol (e.g. Fe)")
    gen_parser.add_argument(
        "--structure",
        required=True,
        choices=["bcc", "hcp", "BCC", "HCP"],
        help="Crystal structure type",
    )
    gen_parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        type=int,
        help="Layer counts to generate",
    )
    gen_parser.add_argument(
        "--vacuum",
        type=float,
        default=25.0,
        help="Vacuum spacing added to the c-axis (default: 25 Å)",
    )
    gen_parser.add_argument(
        "--output",
        type=Path,
        default=Path.cwd(),
        help="Directory to create layer folders in (default: current directory)",
    )
    gen_parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Generate inputs without submitting jobs",
    )

    bonds_parser = subparsers.add_parser("analyze-bonds", help="Analyze bond lengths in POSCAR files")
    bonds_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="POSCAR file or directory containing POSCAR-like files",
    )
    bonds_parser.add_argument(
        "--pattern",
        default="*.vasp",
        help="Glob pattern for files when --input is a directory (default: *.vasp)",
    )
    bonds_parser.add_argument(
        "--max-distance",
        type=float,
        default=3.0,
        help="Maximum bond length to consider in Å (default: 3.0)",
    )
    bonds_parser.add_argument(
        "--output",
        type=Path,
        default=Path("results.dat"),
        help="Output file for analysis summary (default: results.dat)",
    )

    elf_parser = subparsers.add_parser("analyze-elf", help="Analyze ELFCAR files")
    group = elf_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Single ELFCAR file to analyze")
    group.add_argument(
        "--directory",
        type=Path,
        help="Directory containing ELFCAR_* files",
    )
    elf_parser.add_argument(
        "--prefix",
        default="ELFCAR_",
        help="Filename prefix when using --directory (default: ELFCAR_)",
    )
    elf_parser.add_argument(
        "--data-output",
        type=Path,
        default=Path("elfcar_data.dat"),
        help="Output file for ELFCAR metrics when using --directory",
    )
    elf_parser.add_argument(
        "--coords-output",
        type=Path,
        default=Path("elfcar_coords.dat"),
        help="Output file for ELFCAR coordinate summaries when using --directory",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
    except Exception as exc:  # pragma: no cover - CLI error path
        parser.error(str(exc))

    if args.command == "generate-layers":
        return _handle_generate_layers(settings, args)
    if args.command == "analyze-bonds":
        return _handle_analyze_bonds(args)
    if args.command == "analyze-elf":
        return _handle_analyze_elf(args)

    parser.error("Unknown command")
    return 1


def _handle_generate_layers(settings: Settings, args: argparse.Namespace) -> int:
    generator = LayerGenerator(settings, base_directory=args.output)
    request = LayerGenerationRequest(
        element=args.element,
        structure_type=args.structure,
        layer_counts=args.layers,
        vacuum_space=args.vacuum,
        submit_jobs=not args.no_submit,
    )

    created = generator.run(request)
    for path in created:
        print(f"Created layer directory: {path}")
    return 0


def _handle_analyze_bonds(args: argparse.Namespace) -> int:
    input_path: Path = args.input
    files: Iterable[Path]
    if input_path.is_dir():
        files = sorted(input_path.glob(args.pattern))
    else:
        files = [input_path]

    if not files:
        print("No files matched for bond analysis", file=sys.stderr)
        return 1

    lines: list[str] = []
    for file_path in files:
        result = analyze_poscar(file_path, max_distance=args.max_distance)
        lines.extend(_format_bond_result(file_path, result))
        lines.append("\n" + "-" * 40 + "\n")

    args.output.write_text("".join(lines), encoding="utf-8")
    print(f"Bond analysis written to {args.output}")
    return 0


def _handle_analyze_elf(args: argparse.Namespace) -> int:
    if args.file:
        metrics = analyze_elfcar(args.file)
        print(json.dumps(metrics.__dict__, indent=2))
        return 0

    results = analyze_directory(args.directory, prefix=args.prefix)
    if not results:
        print("No ELFCAR files found", file=sys.stderr)
        return 1

    data_lines = ["Layers\tMaxELF\tDist\tAvgELF\n"]
    coord_lines = ["Layers\tMaxFracCoord\tMaxCartCoord\n"]

    for item in results:
        metrics = item.metrics
        data_lines.append(
            f"{item.label}\t{metrics.max_elf:.5f}\t{metrics.shortest_distance:.5f}\t{metrics.average_elf:.5f}\n"
        )
        frac = "\t".join(f"{value:.5f}" for value in metrics.max_frac_coord)
        cart = "\t".join(f"{value:.5f}" for value in metrics.max_cart_coord)
        coord_lines.append(f"{item.label}\t{frac}\t{cart}\n")

    args.data_output.write_text("".join(data_lines), encoding="utf-8")
    args.coords_output.write_text("".join(coord_lines), encoding="utf-8")
    print(f"ELF metrics written to {args.data_output} and {args.coords_output}")
    return 0


def _format_bond_result(file_path: Path, result: BondAnalysisResult) -> list[str]:
    lines = [f"File: {file_path.name}\n", f"Number of Layers: {result.num_layers}\n\n"]

    def _format_section(title: str, summaries) -> None:
        lines.append(f"{title}:\n")
        for summary in summaries:
            lines.append(
                f"{summary.bond_type}: {summary.length:.3f} Angstrom, Count: {summary.count}\n"
            )
        lines.append("\n")

    _format_section("Unique in-plane bonds (unit cell)", result.unit_cell_in_plane)
    _format_section("Unique interlayer bonds (unit cell)", result.unit_cell_interlayer)
    _format_section("Unique in-plane bonds (primitive cell)", result.primitive_in_plane)
    _format_section("Unique interlayer bonds (primitive cell)", result.primitive_interlayer)
    _format_section("Unique in-plane bonds (supercell)", result.supercell_in_plane)
    _format_section("Unique interlayer bonds (supercell)", result.supercell_interlayer)

    return lines


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
