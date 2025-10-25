"""ELF analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from pymatgen.io.vasp.outputs import Elfcar


@dataclass
class ElfMetrics:
    """Metrics computed from an ELFCAR dataset."""

    max_elf: float
    max_frac_coord: Sequence[float]
    max_cart_coord: Sequence[float]
    shortest_distance: float
    average_elf: float


@dataclass
class ElfLayerResult:
    """ELF metrics associated with a labelled layer."""

    label: str
    metrics: ElfMetrics


def analyze_elfcar(path: Path) -> ElfMetrics:
    """Compute ELF metrics for a single ELFCAR file."""

    elfcar = Elfcar.from_file(path)
    elf_data = elfcar.data["total"]
    grid_dimensions = elf_data.shape

    lattice_vectors, atomic_positions_frac = _read_elfcar_origin(path)

    max_elf_value = float(elf_data.max())
    max_index = np.unravel_index(elf_data.argmax(), grid_dimensions)
    max_frac_coord = _index_to_frac(max_index, grid_dimensions)
    max_cart_coord = _frac_to_cart(np.array(max_frac_coord), lattice_vectors)

    atomic_positions_cart = _frac_to_cart(atomic_positions_frac, lattice_vectors)
    distances = np.linalg.norm(atomic_positions_cart - max_cart_coord, axis=1)
    shortest_distance = float(np.min(distances))
    average_elf = float(np.mean(elf_data))

    return ElfMetrics(
        max_elf=max_elf_value,
        max_frac_coord=list(np.round(max_frac_coord, 5)),
        max_cart_coord=list(np.round(max_cart_coord, 5)),
        shortest_distance=round(shortest_distance, 5),
        average_elf=round(average_elf, 5),
    )


def analyze_directory(directory: Path, *, prefix: str = "ELFCAR_") -> List[ElfLayerResult]:
    """Analyze all ELFCAR files in ``directory`` matching the given prefix."""

    candidates = list(directory.glob(f"{prefix}*"))
    labelled = [(_label_for_path(path, prefix), path) for path in candidates]

    def _sort_key(pair: tuple[str, Path]):
        label, _ = pair
        try:
            return (0, int(label))
        except ValueError:
            return (1, label)

    results: list[ElfLayerResult] = []
    for label, path in sorted(labelled, key=_sort_key):
        metrics = analyze_elfcar(path)
        results.append(ElfLayerResult(label=label, metrics=metrics))
    return results


def _label_for_path(path: Path, prefix: str) -> str:
    stem = path.name.replace(prefix, "", 1)
    if stem.lower() == "bulk" or stem == "bulk":
        return "bulk"
    return stem.split(".")[0]


def _read_elfcar_origin(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    lattice_vectors = np.array([list(map(float, lines[i].split())) for i in range(2, 5)])
    num_atoms = int(lines[6].split()[0])
    atomic_positions_frac = np.array(
        [list(map(float, lines[i].split())) for i in range(8, 8 + num_atoms)]
    )
    return lattice_vectors, atomic_positions_frac


def _index_to_frac(index: Sequence[int], grid_dimensions: Sequence[int]) -> np.ndarray:
    return np.array([i / (dim - 1) for i, dim in zip(index, grid_dimensions)], dtype=float)


def _frac_to_cart(frac_coords: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    return np.dot(frac_coords, lattice_vectors)
