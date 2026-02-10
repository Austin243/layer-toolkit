from pymatgen.core import Lattice, Structure

from layer_toolkit.analysis import bonds
from layer_toolkit.analysis.bonds import analyze_structure


def test_count_layers_detects_large_vacuum_gaps() -> None:
    structure = Structure(
        Lattice.from_parameters(3.0, 3.0, 20.0, 90, 90, 90),
        ["Fe", "Fe", "Fe"],
        [
            [0.0, 0.0, 0.10],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.70],
        ],
    )

    layer_count = bonds._count_layers(structure, vacuum_direction=2, gap_threshold=1.5)
    assert layer_count == 3


def test_resolve_bond_key_merges_close_lengths() -> None:
    bucket = {("interlayer", 2.000): 1}
    key = bonds._resolve_bond_key(bucket, "interlayer", 2.006, tolerance=0.008)

    assert key == ("interlayer", 2.000)


def test_analyze_structure_returns_interlayer_bonds_for_layered_geometry() -> None:
    structure = Structure(
        Lattice.from_parameters(3.0, 3.0, 20.0, 90, 90, 90),
        ["Fe", "Fe"],
        [
            [0.25, 0.25, 0.50],
            [0.25, 0.25, 0.60],
        ],
    )

    result = analyze_structure(structure, max_distance=3.0)

    assert result.num_layers == 2
    assert len(result.unit_cell_interlayer) >= 1
    assert all(summary.bond_type == "interlayer" for summary in result.unit_cell_interlayer)
