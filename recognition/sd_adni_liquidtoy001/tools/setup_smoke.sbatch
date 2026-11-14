#!/bin/bash
#SBATCH --job-name=sd-smoke-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_setup_%j.out
#SBATCH --error=logs/smoke_setup_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
base_python="$HOME/miniconda3/envs/torch/bin/python"
"$base_python" -c 'import shutil; assert shutil.disk_usage(".").free > 6*1024**3, "Need 6 GiB free"'
if [[ ! -x .venv-train/bin/python ]]; then
    "$base_python" -m venv --system-site-packages .venv-train
fi
.venv-train/bin/python -m pip install --dry-run -r requirements-smoke.txt
.venv-train/bin/python -m pip install -r requirements-smoke.txt
.venv-train/bin/python -m pip check
.venv-train/bin/python -m pip freeze > "logs/smoke_environment_$SLURM_JOB_ID.txt"
.venv-train/bin/python tools/download_smoke.py
