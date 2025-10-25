#!/bin/bash
#SBATCH --job-name={job_name}
{scheduler_directives}#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --export={export_env}
#SBATCH --output={stdout}
#SBATCH --error={stderr}

ulimit -s unlimited
ulimit -c unlimited
ulimit -d unlimited

mpirun {vasp_executable} >> log
