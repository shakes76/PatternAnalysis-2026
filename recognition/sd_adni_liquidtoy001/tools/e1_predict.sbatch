#!/bin/bash
#SBATCH --job-name=sd-e1-cohort
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/e1_predict_%j.out
#SBATCH --error=logs/e1_predict_%j.err
# S3 of docs/e1-protocol.md: the 64-image cohort from given weights.
# Usage: sbatch tools/e1_predict.sbatch outputs/e1-<job>/weights/unet-ema-epoch-10.safetensors
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
weights="${1:?weights file required}"
.venv-train/bin/python predict.py --weights "$weights" --output "outputs/e1-cohort-$SLURM_JOB_ID"
