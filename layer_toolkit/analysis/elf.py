"""ELF analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

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
class ElfHotspot:
    """A high-value ELF hotspot location and nearest-atom distance."""

    rank: int
    elf_value: float
    frac_coord: Sequence[float]
    cart_coord: Sequence[float]
    shortest_distance: float


@dataclass
class ElfAnalysisResult:
    """Combined ELF metrics and hotspot table for one ELFCAR file."""

    metrics: ElfMetrics
    hotspots: tuple[ElfHotspot, ...]


@dataclass
class ElfLayerResult:
    """ELF metrics associated with a labelled layer."""

    label: str
    metrics: ElfMetrics
    hotspots: tuple[ElfHotspot, ...]


def analyze_elfcar(path: Path) -> ElfMetrics:
    """Compute ELF metrics for a single ELFCAR file."""

    return analyze_elfcar_with_hotspots(path, top_n=1, min_separation_frac=0.0).metrics


def analyze_elfcar_hotspots(
    path: Path,
    *,
    top_n: int = 3,
    min_separation_frac: float = 0.05,
) -> list[ElfHotspot]:
    """Return the top-N ELF hotspots for a single ELFCAR file."""

    result = analyze_elfcar_with_hotspots(
        path,
        top_n=top_n,
        min_separation_frac=min_separation_frac,
    )
    return list(result.hotspots)


def analyze_elfcar_with_hotspots(
    path: Path,
    *,
    top_n: int = 1,
    min_separation_frac: float = 0.05,
) -> ElfAnalysisResult:
    """Compute ELF metrics plus top-N hotspots for a single ELFCAR file."""

    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if min_separation_frac < 0:
        raise ValueError("min_separation_frac must be >= 0")

    elf_data, grid_dimensions, lattice_vectors, atomic_positions_cart = _load_elf_context(path)
    average_elf = round(float(np.mean(elf_data)), 5)
    hotspots = _extract_hotspots(
        elf_data=elf_data,
        grid_dimensions=grid_dimensions,
        lattice_vectors=lattice_vectors,
        atomic_positions_cart=atomic_positions_cart,
        top_n=top_n,
        min_separation_frac=min_separation_frac,
    )
    if not hotspots:
        raise RuntimeError(f"No ELF hotspots could be determined for {path}")

    top = hotspots[0]
    metrics = ElfMetrics(
        max_elf=top.elf_value,
        max_frac_coord=top.frac_coord,
        max_cart_coord=top.cart_coord,
        shortest_distance=top.shortest_distance,
        average_elf=average_elf,
    )
    return ElfAnalysisResult(metrics=metrics, hotspots=tuple(hotspots))


def analyze_directory(
    directory: Path,
    *,
    prefix: str = "ELFCAR_",
    top_n: int = 1,
    min_separation_frac: float = 0.05,
) -> List[ElfLayerResult]:
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
        analysis = analyze_elfcar_with_hotspots(
            path,
            top_n=top_n,
            min_separation_frac=min_separation_frac,
        )
        results.append(
            ElfLayerResult(label=label, metrics=analysis.metrics, hotspots=analysis.hotspots)
        )
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
    atom_counts = [int(value) for value in lines[6].split()]
    num_atoms = sum(atom_counts)
    coordinates_line = 8
    if lines[7].strip().lower().startswith("selective"):
        coordinates_line = 9
    atomic_positions_frac = np.array(
        [list(map(float, lines[i].split()[:3])) for i in range(coordinates_line, coordinates_line + num_atoms)]
    )
    return lattice_vectors, atomic_positions_frac


def _index_to_frac(index: Sequence[int], grid_dimensions: Sequence[int]) -> np.ndarray:
    frac_coords = []
    for value, dim in zip(index, grid_dimensions):
        denominator = (dim - 1) if dim > 1 else 1
        frac_coords.append(value / denominator)
    return np.array(frac_coords, dtype=float)


def _frac_to_cart(frac_coords: np.ndarray, lattice_vectors: np.ndarray) -> np.ndarray:
    return np.dot(frac_coords, lattice_vectors)


def _load_elf_context(path: Path):
    elfcar = Elfcar.from_file(path)
    elf_data = elfcar.data["total"]
    lattice_vectors, atomic_positions_frac = _read_elfcar_origin(path)
    atomic_positions_cart = _frac_to_cart(atomic_positions_frac, lattice_vectors)
    return elf_data, elf_data.shape, lattice_vectors, atomic_positions_cart


def _extract_hotspots(
    *,
    elf_data: np.ndarray,
    grid_dimensions: Sequence[int],
    lattice_vectors: np.ndarray,
    atomic_positions_cart: np.ndarray,
    top_n: int,
    min_separation_frac: float,
) -> list[ElfHotspot]:
    flat = elf_data.reshape(-1)
    if flat.size == 0:
        return []

    candidate_count = min(flat.size, max(top_n * 256, top_n))
    processed: set[int] = set()
    accepted_frac: list[np.ndarray] = []
    hotspots: list[ElfHotspot] = []

    while len(hotspots) < top_n:
        candidate_indices = np.argpartition(flat, -candidate_count)[-candidate_count:]
        ordered_indices = candidate_indices[np.argsort(flat[candidate_indices])[::-1]]

        for raw_index in ordered_indices:
            flat_index = int(raw_index)
            if flat_index in processed:
                continue
            processed.add(flat_index)

            grid_index = np.unravel_index(flat_index, grid_dimensions)
            frac_coord = _index_to_frac(grid_index, grid_dimensions)
            if not _is_far_enough_frac(frac_coord, accepted_frac, min_separation_frac):
                continue

            cart_coord = _frac_to_cart(frac_coord, lattice_vectors)
            if len(atomic_positions_cart) == 0:
                shortest_distance = float("nan")
            else:
                distances = np.linalg.norm(atomic_positions_cart - cart_coord, axis=1)
                shortest_distance = float(np.min(distances))

            accepted_frac.append(frac_coord)
            hotspots.append(
                ElfHotspot(
                    rank=len(hotspots) + 1,
                    elf_value=round(float(flat[flat_index]), 5),
                    frac_coord=list(np.round(frac_coord, 5)),
                    cart_coord=list(np.round(cart_coord, 5)),
                    shortest_distance=round(shortest_distance, 5),
                )
            )
            if len(hotspots) >= top_n:
                return hotspots

        if candidate_count >= flat.size:
            break
        candidate_count = min(flat.size, candidate_count * 2)

    return hotspots


def _is_far_enough_frac(
    candidate: np.ndarray,
    accepted: Sequence[np.ndarray],
    min_separation_frac: float,
) -> bool:
    for existing in accepted:
        if _fractional_distance(candidate, existing) < min_separation_frac:
            return False
    return True


def _fractional_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = np.abs(a - b)
    delta = np.minimum(delta, 1.0 - delta)
    return float(np.linalg.norm(delta))
