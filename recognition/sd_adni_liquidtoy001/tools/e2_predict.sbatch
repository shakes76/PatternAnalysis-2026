#!/bin/bash
#SBATCH --job-name=sd-e2-cohort
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/e2_predict_%j.out
#SBATCH --error=logs/e2_predict_%j.err
# S3 of docs/e2-protocol.md: the 64-image cohort from given weights.
# Usage: sbatch tools/e2_predict.sbatch outputs/e2-<job>/weights/unet-ema-epoch-10.safetensors
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
weights="${1:?weights file required}"
.venv-train/bin/python predict.py --weights "$weights" --output "outputs/e2-cohort-$SLURM_JOB_ID"
