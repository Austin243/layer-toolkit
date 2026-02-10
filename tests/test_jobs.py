import stat
from pathlib import Path

from layer_toolkit.config import SchedulerConfig, Settings, TemplateConfig, ToolPaths
from layer_toolkit.generation.jobs import JobRenderConfig, render_job_script, write_job_script


def _make_settings(tmp_path: Path) -> Settings:
    template_path = tmp_path / "job_template.sh"
    template_path.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "#SBATCH --job-name={job_name}",
                "{scheduler_directives}#SBATCH --nodes={nodes}",
                "#SBATCH --ntasks-per-node={ntasks_per_node}",
                "#SBATCH --export={export_env}",
                "#SBATCH --output={stdout}",
                "#SBATCH --error={stderr}",
                "mpirun {vasp_executable}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return Settings(
        materials_project_api_key=None,
        tools=ToolPaths(
            potcar_root=(tmp_path / "potcars"),
            vasp_std_executable="/opt/vasp/vasp_std",
        ),
        scheduler=SchedulerConfig(
            submit_command="sbatch",
            partition="main",
            exclude="node[1-2]",
            nodes=2,
            ntasks_per_node=24,
            export_env="ALL",
            extra_lines=("#SBATCH --time=01:00:00",),
        ),
        templates=TemplateConfig(
            job_script=str(template_path),
            relax_incar=str(tmp_path / "relax.in"),
            scf_incar=str(tmp_path / "scf.in"),
        ),
    )


def test_render_job_script_includes_scheduler_fields(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    content = render_job_script(settings, JobRenderConfig(job_name="Fe_BCC_3"))

    assert "#SBATCH --job-name=Fe_BCC_3" in content
    assert "#SBATCH --partition=main" in content
    assert "#SBATCH --exclude=node[1-2]" in content
    assert "#SBATCH --time=01:00:00" in content
    assert "#SBATCH --nodes=2" in content
    assert "#SBATCH --ntasks-per-node=24" in content
    assert "mpirun /opt/vasp/vasp_std" in content


def test_write_job_script_sets_execute_permissions(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    output_path = tmp_path / "job.pbs"

    written = write_job_script(
        settings,
        JobRenderConfig(job_name="Fe_BCC_2"),
        output_path,
    )

    assert written == output_path
    mode = stat.S_IMODE(output_path.stat().st_mode)
    assert mode == 0o750
