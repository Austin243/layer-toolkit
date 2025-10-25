"""Layer generation workflows."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence
import logging
import subprocess

import numpy as np
from mp_api.client import MPRester
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from ..config import Settings
from ..io.resources import read_text
from .jobs import JobRenderConfig, write_job_script

_LOGGER = logging.getLogger(__name__)

_STRUCTURE_SPACEGROUPS = {"BCC": 229, "HCP": 194}
_POTCAR_SUFFIXES = ("_pv", "_sv", "", "_s")


@dataclass
class LayerGenerationRequest:
    """Parameters describing the layer workflow user request."""

    element: str
    structure_type: str
    layer_counts: Sequence[int]
    vacuum_space: float = 25.0
    submit_jobs: bool = True


class LayerGenerator:
    """Generate layered structures, input files, and job scripts."""

    def __init__(self, settings: Settings, base_directory: Path | None = None) -> None:
        self.settings = settings
        self.base_directory = Path(base_directory or Path.cwd())
        self.base_directory.mkdir(parents=True, exist_ok=True)

    # API surface ---------------------------------------------------------
    def run(self, request: LayerGenerationRequest) -> list[Path]:
        """Create all layer directories and return their paths."""

        created_paths: list[Path] = []
        structure_type = request.structure_type.upper()
        if structure_type not in _STRUCTURE_SPACEGROUPS:
            raise ValueError(f"Unsupported structure type: {request.structure_type}")

        seen: set[int] = set()
        for layer_count in sorted(request.layer_counts):
            if layer_count < 1:
                raise ValueError("Layer count must be positive")
            if layer_count in seen:
                _LOGGER.debug("Skipping duplicate layer count: %s", layer_count)
                continue
            seen.add(layer_count)
            layer_path = self._prepare_layer(
                element=request.element,
                structure_type=structure_type,
                layer_count=layer_count,
                vacuum_space=request.vacuum_space,
                submit=request.submit_jobs,
            )
            created_paths.append(layer_path)

        return created_paths

    # Internal helpers ----------------------------------------------------
    def _prepare_layer(
        self,
        *,
        element: str,
        structure_type: str,
        layer_count: int,
        vacuum_space: float,
        submit: bool,
    ) -> Path:
        layer_root = self.base_directory / str(layer_count)
        relax_dir = layer_root / "relax"
        scf_dir = layer_root / "scf"
        relax_dir.mkdir(parents=True, exist_ok=True)
        scf_dir.mkdir(parents=True, exist_ok=True)

        _LOGGER.debug("Preparing directories: %s, %s", relax_dir, scf_dir)

        # Assemble POTCAR
        potcar_content = build_potcar([element], self.settings)
        (relax_dir / "POTCAR").write_text(potcar_content, encoding="utf-8")
        (scf_dir / "POTCAR").write_text(potcar_content, encoding="utf-8")

        # INCAR files
        relax_incar = read_text(self.settings.templates.relax_incar)
        scf_incar = read_text(self.settings.templates.scf_incar)
        (relax_dir / "INCAR").write_text(relax_incar, encoding="utf-8")
        (scf_dir / "INCAR").write_text(scf_incar, encoding="utf-8")

        # POSCAR
        structure = build_layer_structure(
            element=element,
            structure_type=structure_type,
            layer_count=layer_count,
            vacuum_space=vacuum_space,
            api_key=self.settings.materials_project_api_key,
        )
        Poscar(structure).write_file(relax_dir / "POSCAR")

        # Job scripts
        job_name = f"{element}_{structure_type}_{layer_count}"
        render_config = JobRenderConfig(job_name=job_name)
        job_script_path = write_job_script(self.settings, render_config, relax_dir / "job.pbs")
        write_job_script(self.settings, render_config, scf_dir / "job.pbs")

        if submit:
            submit_job(job_script_path, self.settings)

        return layer_root


def build_potcar(elements: Iterable[str], settings: Settings) -> str:
    """Concatenate POTCAR entries for the provided elements."""

    potcar_root = settings.tools.potcar_root
    contents: list[str] = []
    for element in elements:
        potcar_path = _locate_potcar(element, potcar_root)
        contents.append(potcar_path.read_text(encoding="utf-8"))

    return "".join(contents)


def _locate_potcar(element: str, potcar_root: Path) -> Path:
    for suffix in _POTCAR_SUFFIXES:
        candidate = potcar_root / f"{element}{suffix}" / "POTCAR"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"POTCAR not found for element {element} in {potcar_root}")


def build_layer_structure(
    *,
    element: str,
    structure_type: str,
    layer_count: int,
    vacuum_space: float,
    api_key: str | None,
) -> Structure:
    """Fetch the base structure and build a layered geometry."""

    if api_key is None:
        raise RuntimeError(
            "Materials Project API key is required; set it in config.json or MP_API_KEY"
        )

    spacegroup_number = _STRUCTURE_SPACEGROUPS[structure_type]
    with MPRester(api_key) as mpr:
        docs = mpr.summary.search(
            elements=[element],
            spacegroup_number=spacegroup_number,
            fields=["structure"],
        )
    if not docs:
        raise ValueError(f"No {structure_type} structure found for {element}")

    initial_structure: Structure = docs[0].structure

    atom_indices = range(len(initial_structure))
    distances = [
        initial_structure.get_distance(i, j)
        for i, j in combinations(atom_indices, 2)
    ]
    average_bond_distance = float(np.mean(distances)) if distances else 1.0

    lattice = initial_structure.lattice
    if layer_count == 1:
        new_c = lattice.c + vacuum_space
    else:
        new_c = average_bond_distance * (layer_count - 1) + vacuum_space

    new_lattice = Lattice([[lattice.a, 0, 0], [0, lattice.b, 0], [0, 0, new_c]])

    frac_coords = []
    for idx in range(layer_count):
        z_coord = (average_bond_distance * idx + vacuum_space / 2.0) / new_c
        if structure_type == "BCC":
            ab_coords = [0.25, 0.75] if idx % 2 == 0 else [0.75, 0.25]
        else:  # HCP
            ab_coords = [0.0, 0.0] if idx % 2 == 0 else [2 / 3, 1 / 3]
        frac_coords.append([ab_coords[0], ab_coords[1], z_coord])

    species = [element] * layer_count
    return Structure(new_lattice, species, frac_coords)


def submit_job(job_script_path: Path, settings: Settings) -> None:
    """Submit the generated job script using the configured scheduler command."""

    submit_cmd = settings.scheduler.submit_command
    cmd = [submit_cmd, str(job_script_path)]
    _LOGGER.info("Submitting job: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Submit command not found: {submit_cmd}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Job submission failed: {exc}") from exc
