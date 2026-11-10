#!/bin/bash
#SBATCH --job-name=sd-e1-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/e1_setup_%j.out
#SBATCH --error=logs/e1_setup_%j.err
# Adds the E1 plotting dependency to the existing training overlay environment.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
.venv-train/bin/python -m pip install --dry-run -r requirements-train.txt
.venv-train/bin/python -m pip install -r requirements-train.txt
.venv-train/bin/python -m pip check
.venv-train/bin/python -m pip freeze > "logs/e1_environment_$SLURM_JOB_ID.txt"
.venv-train/bin/python -c 'import matplotlib, diffusers, peft; print(matplotlib.__version__, diffusers.__version__, peft.__version__)'
