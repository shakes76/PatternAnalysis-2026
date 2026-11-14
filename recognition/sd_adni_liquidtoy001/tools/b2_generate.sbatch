#!/bin/bash
#SBATCH --job-name=sd-b2-cohort
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/b2_generate_%j.out
#SBATCH --error=logs/b2_generate_%j.err
# S3 of docs/b2-gan-protocol.md: the 64 fixed seeds from the result checkpoint.
# Usage: sbatch tools/b2_generate.sbatch outputs/b2-<job>/final.pt
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
checkpoint="${1:?checkpoint required}"
.venv-train/bin/python tools/baseline_gan.py generate --checkpoint "$checkpoint" \
    --output "outputs/b2-cohort-$SLURM_JOB_ID"
