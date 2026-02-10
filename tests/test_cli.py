import argparse
from pathlib import Path

import pytest

from layer_toolkit import cli
from layer_toolkit.analysis.elf import ElfHotspot, ElfLayerResult, ElfMetrics
from layer_toolkit.config import Settings, ToolPaths


def _minimal_settings(tmp_path: Path) -> Settings:
    return Settings(
        materials_project_api_key="api-key",
        tools=ToolPaths(
            potcar_root=(tmp_path / "potcars"),
            vasp_std_executable="/usr/bin/vasp_std",
        ),
    )


def test_build_parser_parses_new_generate_layer_options() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "generate-layers",
            "--element",
            "Fe",
            "--structure",
            "bcc",
            "--layers",
            "2",
            "4",
            "--material-id",
            "mp-13",
            "--require-stable",
            "--max-energy-above-hull",
            "0.02",
        ]
    )

    assert args.material_id == "mp-13"
    assert args.require_stable is True
    assert args.max_energy_above_hull == pytest.approx(0.02)


def test_handle_generate_layers_threads_new_selection_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    class _FakeGenerator:
        def __init__(self, settings, base_directory):
            self.settings = settings
            self.base_directory = base_directory

        def run(self, request):
            captured["request"] = request
            return [Path("1")]

    monkeypatch.setattr(cli, "LayerGenerator", _FakeGenerator)
    args = argparse.Namespace(
        element="Fe",
        structure="bcc",
        layers=[1],
        vacuum=25.0,
        no_submit=False,
        output=tmp_path,
        material_id="mp-13",
        require_stable=True,
        max_energy_above_hull=0.01,
    )

    code = cli._handle_generate_layers(_minimal_settings(tmp_path), args)
    request = captured["request"]

    assert code == 0
    assert request.material_id == "mp-13"
    assert request.require_stable is True
    assert request.max_energy_above_hull == pytest.approx(0.01)


def test_handle_analyze_elf_rejects_invalid_top_n() -> None:
    args = argparse.Namespace(
        file=Path("ELFCAR"),
        directory=None,
        prefix="ELFCAR_",
        data_output=Path("elfcar_data.dat"),
        coords_output=Path("elfcar_coords.dat"),
        hotspots_output=Path("elfcar_hotspots.dat"),
        top_n=0,
        min_separation_frac=0.05,
    )

    assert cli._handle_analyze_elf(args) == 1


def test_handle_analyze_elf_directory_writes_hotspot_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metrics = ElfMetrics(
        max_elf=0.95,
        max_frac_coord=[0.1, 0.2, 0.3],
        max_cart_coord=[1.0, 2.0, 3.0],
        shortest_distance=0.42,
        average_elf=0.25,
    )
    hotspots = (
        ElfHotspot(
            rank=1,
            elf_value=0.95,
            frac_coord=[0.1, 0.2, 0.3],
            cart_coord=[1.0, 2.0, 3.0],
            shortest_distance=0.42,
        ),
        ElfHotspot(
            rank=2,
            elf_value=0.90,
            frac_coord=[0.4, 0.5, 0.6],
            cart_coord=[4.0, 5.0, 6.0],
            shortest_distance=0.55,
        ),
    )

    def _fake_analyze_directory(directory: Path, *, prefix: str, top_n: int, min_separation_frac: float):
        return [ElfLayerResult(label="2", metrics=metrics, hotspots=hotspots)]

    monkeypatch.setattr(cli, "analyze_directory", _fake_analyze_directory)

    args = argparse.Namespace(
        file=None,
        directory=tmp_path,
        prefix="ELFCAR_",
        data_output=tmp_path / "data.dat",
        coords_output=tmp_path / "coords.dat",
        hotspots_output=tmp_path / "hotspots.dat",
        top_n=2,
        min_separation_frac=0.1,
    )

    code = cli._handle_analyze_elf(args)
    assert code == 0
    assert args.hotspots_output.exists()
    contents = args.hotspots_output.read_text(encoding="utf-8")
    assert "Layers\tRank\tELF\tDist\tFracCoord\tCartCoord" in contents
    assert "2\t1\t0.95000\t0.42000" in contents
