"""Bond analysis helpers for layered structures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


@dataclass
class BondSummary:
    """Summary of a unique bond length and its frequency."""

    bond_type: str
    length: float
    count: int


@dataclass
class BondAnalysisResult:
    """Statistics derived from bond analysis."""

    num_layers: int
    unit_cell_in_plane: tuple[BondSummary, ...]
    unit_cell_interlayer: tuple[BondSummary, ...]
    primitive_in_plane: tuple[BondSummary, ...]
    primitive_interlayer: tuple[BondSummary, ...]
    supercell_in_plane: tuple[BondSummary, ...]
    supercell_interlayer: tuple[BondSummary, ...]


def analyze_poscar(path: Path, *, max_distance: float = 3.0) -> BondAnalysisResult:
    """Analyze bond statistics for a POSCAR/CONTCAR file."""

    structure = Poscar.from_file(path).structure
    return analyze_structure(structure, max_distance=max_distance)


def analyze_structure(structure: Structure, *, max_distance: float = 3.0) -> BondAnalysisResult:
    """Perform bond analysis for the provided structure."""

    vacuum_direction = int(np.argmax([np.linalg.norm(v) for v in structure.lattice.matrix]))
    num_layers = _count_layers(structure, vacuum_direction)

    unit_plane, unit_inter = _collect_bonds(structure, max_distance, vacuum_direction)

    primitive_structure = SpacegroupAnalyzer(structure).get_primitive_standard_structure()
    prim_plane, prim_inter = _collect_bonds(primitive_structure, max_distance, vacuum_direction)

    supercell_structure = _build_supercell(structure, vacuum_direction)
    super_plane, super_inter = _collect_bonds(supercell_structure, max_distance, vacuum_direction)

    return BondAnalysisResult(
        num_layers=num_layers,
        unit_cell_in_plane=_dict_to_summaries(unit_plane),
        unit_cell_interlayer=_dict_to_summaries(unit_inter),
        primitive_in_plane=_dict_to_summaries(prim_plane),
        primitive_interlayer=_dict_to_summaries(prim_inter),
        supercell_in_plane=_dict_to_summaries(super_plane),
        supercell_interlayer=_dict_to_summaries(super_inter),
    )


def _count_layers(structure: Structure, vacuum_direction: int, gap_threshold: float = 1.5) -> int:
    coords = sorted(site.coords[vacuum_direction] for site in structure)
    gaps = sum(1 for a, b in zip(coords, coords[1:]) if (b - a) > gap_threshold)
    return gaps + 1


def _build_supercell(structure: Structure, vacuum_direction: int) -> Structure:
    scaling_matrix = np.diag([3, 3, 3]).astype(int)
    scaling_matrix[vacuum_direction] = [1, 1, 1]

    supercell = structure.copy()
    supercell.make_supercell(scaling_matrix.tolist())
    return supercell


def _collect_bonds(
    structure: Structure,
    max_distance: float,
    vacuum_direction: int,
) -> Tuple[dict[Tuple[str, float], int], dict[Tuple[str, float], int]]:
    in_plane: dict[tuple[str, float], int] = {}
    interlayer: dict[tuple[str, float], int] = {}
    processed: set[tuple[tuple[float, ...], tuple[float, ...]]] = set()

    for site in structure:
        for neighbor_info in structure.get_neighbors(site, max_distance):
            neighbor = neighbor_info[0]
            frac = structure.lattice.get_fractional_coords(neighbor.coords)
            if any(coord < 0 or coord >= 1 for coord in frac):
                continue

            bond_id = tuple(sorted((tuple(site.coords), tuple(neighbor.coords))))
            if bond_id in processed:
                continue
            processed.add(bond_id)

            vector = site.coords - neighbor.coords
            bond_type = "interlayer" if int(np.argmax(np.abs(vector))) == vacuum_direction else "in-plane"
            bucket = interlayer if bond_type == "interlayer" else in_plane

            bond_length = float(np.linalg.norm(vector))
            key = _resolve_bond_key(bucket, bond_type, bond_length)
            bucket[key] = bucket.get(key, 0) + 1

    return in_plane, interlayer


def _dict_to_summaries(data: dict[tuple[str, float], int]) -> tuple[BondSummary, ...]:
    summaries = [BondSummary(bond_type=key[0], length=key[1], count=count) for key, count in data.items()]
    summaries.sort(key=lambda item: item.length)
    return tuple(summaries)


def _resolve_bond_key(bucket: dict[tuple[str, float], int], bond_type: str, bond_length: float, tolerance: float = 0.008) -> tuple[str, float]:
    for key in bucket:
        if key[0] == bond_type and abs(key[1] - bond_length) <= tolerance:
            return key
    return (bond_type, round(bond_length, 3))
