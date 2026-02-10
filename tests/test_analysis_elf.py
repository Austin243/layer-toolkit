from pathlib import Path

import numpy as np
import pytest

from layer_toolkit.analysis import elf
from layer_toolkit.analysis.elf import (
    ElfAnalysisResult,
    ElfHotspot,
    ElfMetrics,
    analyze_directory,
    analyze_elfcar_with_hotspots,
)


def test_read_elfcar_origin_handles_multi_species_and_selective_dynamics(tmp_path: Path) -> None:
    elfcar_path = tmp_path / "ELFCAR_mock"
    elfcar_path.write_text(
        "\n".join(
            [
                "Mock ELFCAR",
                "1.0",
                "3.0 0.0 0.0",
                "0.0 3.0 0.0",
                "0.0 0.0 20.0",
                "Fe O",
                "1 2",
                "Selective dynamics",
                "Direct",
                "0.10 0.10 0.10 T T T",
                "0.20 0.20 0.20 T T T",
                "0.30 0.30 0.30 T T T",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    lattice_vectors, frac_positions = elf._read_elfcar_origin(elfcar_path)
    assert lattice_vectors.shape == (3, 3)
    assert frac_positions.shape == (3, 3)
    assert np.allclose(frac_positions[2], np.array([0.30, 0.30, 0.30]))


def test_analyze_directory_sorts_numeric_then_text_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ["ELFCAR_10", "ELFCAR_2", "ELFCAR_bulk", "ELFCAR_alpha"]:
        (tmp_path / name).write_text("", encoding="utf-8")

    def _fake_analyze(path: Path, *, top_n: int, min_separation_frac: float) -> ElfAnalysisResult:
        metrics = ElfMetrics(
            max_elf=1.0,
            max_frac_coord=[0.0, 0.0, 0.0],
            max_cart_coord=[0.0, 0.0, 0.0],
            shortest_distance=0.5,
            average_elf=0.25,
        )
        hotspots = (
            ElfHotspot(
                rank=1,
                elf_value=1.0,
                frac_coord=[0.0, 0.0, 0.0],
                cart_coord=[0.0, 0.0, 0.0],
                shortest_distance=0.5,
            ),
        )
        return ElfAnalysisResult(metrics=metrics, hotspots=hotspots)

    monkeypatch.setattr(elf, "analyze_elfcar_with_hotspots", _fake_analyze)
    results = analyze_directory(tmp_path, prefix="ELFCAR_", top_n=1, min_separation_frac=0.05)
    labels = [item.label for item in results]
    assert labels == ["2", "10", "alpha", "bulk"]


def test_analyze_elfcar_with_hotspots_validates_inputs() -> None:
    with pytest.raises(ValueError, match="top_n must be >= 1"):
        analyze_elfcar_with_hotspots(Path("ELFCAR"), top_n=0)
    with pytest.raises(ValueError, match="min_separation_frac must be >= 0"):
        analyze_elfcar_with_hotspots(Path("ELFCAR"), min_separation_frac=-0.1)


def test_analyze_elfcar_with_hotspots_uses_highest_hotspot_for_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    elf_data = np.zeros((4, 4, 4))
    elf_data[1, 1, 1] = 0.99
    elf_data[3, 3, 3] = 0.97

    def _fake_context(path: Path):
        lattice_vectors = np.eye(3)
        atomic_cart = np.array([[0.0, 0.0, 0.0]])
        return elf_data, elf_data.shape, lattice_vectors, atomic_cart

    monkeypatch.setattr(elf, "_load_elf_context", _fake_context)
    result = analyze_elfcar_with_hotspots(Path("ELFCAR"), top_n=2, min_separation_frac=0.1)

    assert result.metrics.max_elf == 0.99
    assert len(result.hotspots) == 2
    assert result.hotspots[0].rank == 1
    assert result.hotspots[1].rank == 2
