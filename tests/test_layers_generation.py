from pathlib import Path
import subprocess

import pytest
from pymatgen.core import Lattice, Structure

from layer_toolkit.config import SchedulerConfig, Settings, TemplateConfig, ToolPaths
from layer_toolkit.generation import layers
from layer_toolkit.generation.layers import (
    LayerGenerationRequest,
    LayerGenerator,
    build_layer_structure,
    build_potcar,
    submit_job,
)


def _make_settings(tmp_path: Path) -> Settings:
    template_path = tmp_path / "job_template.sh"
    template_path.write_text(
        "#!/bin/bash\n#SBATCH --job-name={job_name}\n{scheduler_directives}"
        "#SBATCH --nodes={nodes}\n#SBATCH --ntasks-per-node={ntasks_per_node}\n"
        "#SBATCH --export={export_env}\n#SBATCH --output={stdout}\n"
        "#SBATCH --error={stderr}\nmpirun {vasp_executable}\n",
        encoding="utf-8",
    )
    (tmp_path / "relax.in").write_text("SYSTEM = opt\n", encoding="utf-8")
    (tmp_path / "scf.in").write_text("SYSTEM = scf\n", encoding="utf-8")

    potcar_root = tmp_path / "potcars"
    potcar_root.mkdir(parents=True, exist_ok=True)

    return Settings(
        materials_project_api_key="api-key",
        tools=ToolPaths(
            potcar_root=potcar_root.resolve(),
            vasp_std_executable="/usr/bin/vasp_std",
        ),
        scheduler=SchedulerConfig(submit_command="sbatch"),
        templates=TemplateConfig(
            job_script=str(template_path),
            relax_incar=str(tmp_path / "relax.in"),
            scf_incar=str(tmp_path / "scf.in"),
        ),
    )


def test_locate_potcar_prefers_priority_suffixes(tmp_path: Path) -> None:
    potcar_root = tmp_path / "potcars"
    (potcar_root / "Fe" / "POTCAR").parent.mkdir(parents=True, exist_ok=True)
    (potcar_root / "Fe_sv" / "POTCAR").parent.mkdir(parents=True, exist_ok=True)
    (potcar_root / "Fe_pv" / "POTCAR").parent.mkdir(parents=True, exist_ok=True)
    (potcar_root / "Fe" / "POTCAR").write_text("plain", encoding="utf-8")
    (potcar_root / "Fe_sv" / "POTCAR").write_text("sv", encoding="utf-8")
    (potcar_root / "Fe_pv" / "POTCAR").write_text("pv", encoding="utf-8")

    selected = layers._locate_potcar("Fe", potcar_root)
    assert selected == (potcar_root / "Fe_pv" / "POTCAR")


def test_build_potcar_concatenates_per_element_entries(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    (settings.tools.potcar_root / "Fe").mkdir(parents=True, exist_ok=True)
    (settings.tools.potcar_root / "O").mkdir(parents=True, exist_ok=True)
    (settings.tools.potcar_root / "Fe" / "POTCAR").write_text("Fe\n", encoding="utf-8")
    (settings.tools.potcar_root / "O" / "POTCAR").write_text("O\n", encoding="utf-8")

    content = build_potcar(["Fe", "O"], settings)
    assert content == "Fe\nO\n"


def test_layer_generator_deduplicates_and_sorts_layer_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path)
    generator = LayerGenerator(settings, base_directory=tmp_path / "out")
    called: list[int] = []

    def _fake_prepare(
        self,
        *,
        element: str,
        structure_type: str,
        layer_count: int,
        vacuum_space: float,
        submit: bool,
        material_id: str | None,
        require_stable: bool,
        max_energy_above_hull: float | None,
    ) -> Path:
        called.append(layer_count)
        return self.base_directory / str(layer_count)

    monkeypatch.setattr(LayerGenerator, "_prepare_layer", _fake_prepare)

    request = LayerGenerationRequest(
        element="Fe",
        structure_type="bcc",
        layer_counts=[3, 1, 3, 2],
    )
    paths = generator.run(request)

    assert called == [1, 2, 3]
    assert paths == [
        generator.base_directory / "1",
        generator.base_directory / "2",
        generator.base_directory / "3",
    ]


def test_layer_generator_rejects_non_positive_layer_count(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    generator = LayerGenerator(settings, base_directory=tmp_path / "out")
    request = LayerGenerationRequest(element="Fe", structure_type="bcc", layer_counts=[0])

    with pytest.raises(ValueError, match="Layer count must be positive"):
        generator.run(request)


def test_submit_job_reports_missing_submit_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path)
    script_path = tmp_path / "job.pbs"
    script_path.write_text("#!/bin/bash\n", encoding="utf-8")

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(layers.subprocess, "run", _raise_not_found)

    with pytest.raises(RuntimeError, match="Submit command not found"):
        submit_job(script_path, settings)


def test_submit_job_reports_non_zero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _make_settings(tmp_path)
    script_path = tmp_path / "job.pbs"
    script_path.write_text("#!/bin/bash\n", encoding="utf-8")

    def _raise_called_process(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=["sbatch", str(script_path)])

    monkeypatch.setattr(layers.subprocess, "run", _raise_called_process)

    with pytest.raises(RuntimeError, match="Job submission failed"):
        submit_job(script_path, settings)


def test_build_layer_structure_uses_query_filters_and_selected_doc(monkeypatch: pytest.MonkeyPatch) -> None:
    search_calls: list[dict] = []
    structure_low = Structure(Lattice.cubic(3.2), ["Fe"], [[0, 0, 0]])
    structure_high = Structure(Lattice.cubic(4.5), ["Fe"], [[0, 0, 0]])
    docs = [
        {
            "material_id": "mp-high",
            "energy_above_hull": 0.05,
            "is_stable": False,
            "symmetry": {"number": 229},
            "structure": structure_high,
        },
        {
            "material_id": "mp-low",
            "energy_above_hull": 0.01,
            "is_stable": True,
            "symmetry": {"number": 229},
            "structure": structure_low,
        },
    ]

    class _FakeSummary:
        def search(self, **kwargs):
            search_calls.append(kwargs)
            return docs

    class _FakeMPRester:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.summary = _FakeSummary()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(layers, "MPRester", _FakeMPRester)

    result = build_layer_structure(
        element="Fe",
        structure_type="BCC",
        layer_count=3,
        vacuum_space=20.0,
        api_key="api-key",
        material_id="mp-low",
        require_stable=True,
        max_energy_above_hull=0.02,
    )

    assert len(result) == 3
    assert result.lattice.a == pytest.approx(structure_low.lattice.a)
    assert search_calls
    kwargs = search_calls[0]
    assert kwargs["material_ids"] == ["mp-low"]
    assert kwargs["is_stable"] is True
    assert kwargs["energy_above_hull"] == (0.0, 0.02)


def test_build_layer_structure_rejects_negative_hull_filter() -> None:
    with pytest.raises(ValueError, match="max_energy_above_hull must be >= 0"):
        build_layer_structure(
            element="Fe",
            structure_type="BCC",
            layer_count=2,
            vacuum_space=25.0,
            api_key="api-key",
            max_energy_above_hull=-0.1,
        )
