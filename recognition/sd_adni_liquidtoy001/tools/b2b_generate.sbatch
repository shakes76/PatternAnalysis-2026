#!/bin/bash
#SBATCH --job-name=sd-b2b-cohort
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/b2b_generate_%j.out
#SBATCH --error=logs/b2b_generate_%j.err
# S3 of docs/b2b-gan-per-class-protocol.md: 32 fixed seeds from each class's result.
# Usage: sbatch tools/b2b_generate.sbatch AD_CHECKPOINT NC_CHECKPOINT
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
ad="${1:?AD checkpoint required}"
nc="${2:?NC checkpoint required}"
.venv-train/bin/python tools/baseline_gan.py generate --output "outputs/b2b-cohort-$SLURM_JOB_ID" \
    --class-checkpoint "AD=$ad" --class-checkpoint "NC=$nc"
