"""Configuration handling for the Layer Toolkit."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
import json
import os

_DEFAULT_CONFIG_FILENAMES = (
    "layer_toolkit.config.json",
    "config.json",
)


@dataclass
class ToolPaths:
    """File-system paths for external tools and resources."""

    potcar_root: Path
    vasp_std_executable: str
    calypso_executable: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ToolPaths":
        try:
            potcar_root = Path(data["potcar_root"]).expanduser().resolve()
            vasp_std = str(data["vasp_std_executable"])
        except KeyError as exc:  # pragma: no cover - configuration error path
            raise ValueError(f"Missing required tool path setting: {exc.args[0]}") from exc

        calypso_exec = data.get("calypso_executable")

        return cls(
            potcar_root=potcar_root,
            vasp_std_executable=vasp_std,
            calypso_executable=str(calypso_exec) if calypso_exec else None,
        )


@dataclass
class SchedulerConfig:
    """Configuration for the batch scheduler submission."""

    submit_command: str = "qsub"
    partition: Optional[str] = None
    exclude: Optional[str] = None
    nodes: int = 1
    ntasks_per_node: int = 48
    export_env: str = "ALL"
    extra_lines: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SchedulerConfig":
        return cls(
            submit_command=str(data.get("submit_command", "qsub")),
            partition=data.get("partition"),
            exclude=data.get("exclude"),
            nodes=int(data.get("nodes", 1)),
            ntasks_per_node=int(data.get("ntasks_per_node", 48)),
            export_env=str(data.get("export_env", "ALL")),
            extra_lines=tuple(str(line) for line in data.get("extra_lines", [])),
        )


@dataclass
class TemplateConfig:
    """Locations (relative to the config file or package resources) for template files."""

    job_script: str = "resources/job_template.sh"
    relax_incar: str = "resources/incar_relax.in"
    scf_incar: str = "resources/incar_scf.in"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "TemplateConfig":
        return cls(
            job_script=str(data.get("job_script", "resources/job_template.sh")),
            relax_incar=str(data.get("relax_incar", "resources/incar_relax.in")),
            scf_incar=str(data.get("scf_incar", "resources/incar_scf.in")),
        )


@dataclass
class Settings:
    """All runtime settings for the toolkit."""

    materials_project_api_key: Optional[str]
    tools: ToolPaths
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, root: Path) -> "Settings":
        api_key = data.get("materials_project_api_key") or os.getenv("MP_API_KEY")
        tools = ToolPaths.from_mapping(data.get("tools", {}))
        scheduler = SchedulerConfig.from_mapping(data.get("scheduler", {}))
        templates = TemplateConfig.from_mapping(data.get("templates", {}))

        return cls(
            materials_project_api_key=str(api_key) if api_key else None,
            tools=tools,
            scheduler=scheduler,
            templates=_resolve_templates(templates, root=root),
        )


def _resolve_templates(templates: TemplateConfig, *, root: Path) -> TemplateConfig:
    """Resolve any relative template paths against the provided root."""

    def _resolve(path_str: str) -> str:
        path = Path(path_str)
        if not path.is_absolute():
            path = (root / path).resolve()
        return str(path)

    return TemplateConfig(
        job_script=_resolve(templates.job_script),
        relax_incar=_resolve(templates.relax_incar),
        scf_incar=_resolve(templates.scf_incar),
    )


def _read_json_config(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """Load settings from JSON and environment variables."""

    if config_path is not None:
        chosen_path = Path(config_path).expanduser().resolve()
        if not chosen_path.exists():  # pragma: no cover - configuration error path
            raise FileNotFoundError(f"Configuration file not found: {chosen_path}")
        data = _read_json_config(chosen_path)
        return Settings.from_mapping(data, root=chosen_path.parent)

    env_path = os.getenv("LAYER_TOOLKIT_CONFIG")
    if env_path:
        return load_settings(Path(env_path))

    for filename in _DEFAULT_CONFIG_FILENAMES:
        candidate = Path.cwd() / filename
        if candidate.exists():
            data = _read_json_config(candidate)
            return Settings.from_mapping(data, root=candidate.parent)

    raise FileNotFoundError(
        "No configuration file found. Set LAYER_TOOLKIT_CONFIG or create config.json."
    )


__all__ = ["Settings", "ToolPaths", "SchedulerConfig", "TemplateConfig", "load_settings"]
