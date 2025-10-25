"""Job script helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..io.resources import read_text


@dataclass
class JobRenderConfig:
    """Job script rendering parameters."""

    job_name: str
    stdout: str = "%j.out"
    stderr: str = "%j.err"


def render_job_script(settings: Settings, params: JobRenderConfig) -> str:
    """Render the batch submission script using configured templates."""

    template = read_text(settings.templates.job_script)
    scheduler = settings.scheduler

    directives: list[str] = []
    if scheduler.partition:
        directives.append(f"#SBATCH --partition={scheduler.partition}\n")
    if scheduler.exclude:
        directives.append(f"#SBATCH --exclude={scheduler.exclude}\n")
    for extra in scheduler.extra_lines:
        directives.append(f"{extra}\n")

    scheduler_directives = "".join(directives)

    return template.format(
        job_name=params.job_name,
        scheduler_directives=scheduler_directives,
        nodes=scheduler.nodes,
        ntasks_per_node=scheduler.ntasks_per_node,
        export_env=scheduler.export_env,
        stdout=params.stdout,
        stderr=params.stderr,
        vasp_executable=settings.tools.vasp_std_executable,
    )


def write_job_script(settings: Settings, params: JobRenderConfig, destination: Path) -> Path:
    """Write the rendered script to ``destination`` and return the path."""

    content = render_job_script(settings, params)
    destination.write_text(content, encoding="utf-8")
    destination.chmod(0o750)
    return destination
