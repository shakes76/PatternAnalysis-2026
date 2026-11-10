#!/bin/bash
#SBATCH --job-name=sd-b3-cohort
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/b3_predict_%j.out
#SBATCH --error=logs/b3_predict_%j.err
# S3 of docs/b3-ldm-scratch-protocol.md: the 64-image cohort at one guidance value.
# Usage: sbatch tools/b3_predict.sbatch WEIGHTS [GUIDANCE, default 7.5]
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
weights="${1:?weights file required}"
guidance="${2:-7.5}"
.venv-train/bin/python predict.py --unet-init scratch --weights "$weights" \
    --guidance "$guidance" --output "outputs/b3-cohort-$SLURM_JOB_ID"
