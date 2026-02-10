import json
from pathlib import Path

import pytest

from layer_toolkit.config import load_settings


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_config_payload(tmp_path: Path) -> dict:
    template_dir = tmp_path / "templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    (template_dir / "job_template.sh").write_text("#!/bin/bash\n", encoding="utf-8")
    (template_dir / "relax.in").write_text("SYSTEM = opt\n", encoding="utf-8")
    (template_dir / "scf.in").write_text("SYSTEM = scf\n", encoding="utf-8")

    potcar_root = tmp_path / "potcars"
    potcar_root.mkdir(parents=True, exist_ok=True)

    return {
        "materials_project_api_key": "config-key",
        "tools": {
            "potcar_root": str(potcar_root),
            "vasp_std_executable": "/usr/bin/vasp_std",
        },
        "scheduler": {
            "submit_command": "sbatch",
            "partition": "compute",
            "extra_lines": ["#SBATCH --time=01:00:00"],
        },
        "templates": {
            "job_script": "templates/job_template.sh",
            "relax_incar": "templates/relax.in",
            "scf_incar": "templates/scf.in",
        },
    }


def test_load_settings_explicit_path_resolves_relative_templates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    _write_json(config_path, _make_config_payload(tmp_path))

    settings = load_settings(config_path)

    assert settings.tools.potcar_root == (tmp_path / "potcars").resolve()
    assert settings.templates.job_script == str((tmp_path / "templates" / "job_template.sh").resolve())
    assert settings.templates.relax_incar == str((tmp_path / "templates" / "relax.in").resolve())
    assert settings.templates.scf_incar == str((tmp_path / "templates" / "scf.in").resolve())
    assert settings.materials_project_api_key == "config-key"


def test_load_settings_uses_environment_api_key_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _make_config_payload(tmp_path)
    payload.pop("materials_project_api_key", None)
    config_path = tmp_path / "config.json"
    _write_json(config_path, payload)
    monkeypatch.setenv("MP_API_KEY", "env-key")

    settings = load_settings(config_path)

    assert settings.materials_project_api_key == "env-key"


def test_load_settings_discovers_config_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.json"
    _write_json(config_path, _make_config_payload(tmp_path))
    monkeypatch.chdir(tmp_path)

    settings = load_settings()

    assert settings.scheduler.partition == "compute"
    assert settings.scheduler.submit_command == "sbatch"


def test_load_settings_raises_when_no_config_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LAYER_TOOLKIT_CONFIG", raising=False)

    with pytest.raises(FileNotFoundError):
        load_settings()
