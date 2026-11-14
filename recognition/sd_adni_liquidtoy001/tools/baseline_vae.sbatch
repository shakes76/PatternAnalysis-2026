#!/bin/bash
#SBATCH --job-name=baseline-vae
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline_vae_%j.out
#SBATCH --error=logs/baseline_vae_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-train/bin/python tools/baseline_vae.py --output "outputs/baseline-vae-$SLURM_JOB_ID" "$@"
