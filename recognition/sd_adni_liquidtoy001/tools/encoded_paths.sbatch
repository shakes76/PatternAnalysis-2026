#!/bin/bash
#SBATCH --job-name=sd-encoded
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/encoded_%j.out
#SBATCH --error=logs/encoded_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
.venv-train/bin/python tools/encoded_paths.py --output "outputs/encoded-$SLURM_JOB_ID"
