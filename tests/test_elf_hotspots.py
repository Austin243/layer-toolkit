import numpy as np

from layer_toolkit.analysis.elf import _extract_hotspots, _fractional_distance, _index_to_frac


def test_fractional_distance_wraps_periodic_boundaries() -> None:
    a = np.array([0.98, 0.02, 0.50])
    b = np.array([0.01, 0.98, 0.50])

    # Distances wrap around unit-cell boundaries for the first two axes.
    dist = _fractional_distance(a, b)
    assert round(dist, 5) == round(np.sqrt(0.03**2 + 0.04**2), 5)


def test_index_to_frac_handles_singleton_dimensions() -> None:
    frac = _index_to_frac((0, 3, 1), (1, 5, 2))
    assert np.allclose(frac, np.array([0.0, 0.75, 1.0]))


def test_extract_hotspots_respects_min_separation() -> None:
    elf_data = np.zeros((4, 4, 4))
    elf_data[0, 0, 0] = 1.0
    elf_data[0, 0, 1] = 0.99
    elf_data[2, 2, 2] = 0.98

    hotspots = _extract_hotspots(
        elf_data=elf_data,
        grid_dimensions=elf_data.shape,
        lattice_vectors=np.eye(3),
        atomic_positions_cart=np.array([[0.0, 0.0, 0.0]]),
        top_n=2,
        min_separation_frac=0.4,
    )

    assert len(hotspots) == 2
    assert hotspots[0].elf_value == 1.0
    assert hotspots[1].elf_value == 0.98
