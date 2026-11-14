#!/bin/bash
#SBATCH --job-name=sd-vae-setup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/vae_setup_%j.out
#SBATCH --error=logs/vae_setup_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PIP_NO_CACHE_DIR=1 HF_HUB_DISABLE_TELEMETRY=1
base_python="$HOME/miniconda3/envs/torch/bin/python"
if [[ ! -x .venv/bin/python ]]; then
    "$base_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -r requirements-vae.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > logs/vae_environment.txt
.venv/bin/python tools/download_vae.py
