#!/bin/bash
#SBATCH --job-name=sd-r1-cohort
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/r1_predict_%j.out
#SBATCH --error=logs/r1_predict_%j.err
# S3 of docs/r1-seed-replicates-protocol.md: the 64-image cohort from given weights.
# Usage: sbatch tools/r1_predict.sbatch outputs/r1-<model>-<job>/weights/<file>.safetensors
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
weights="${1:?weights file required}"
.venv-train/bin/python predict.py --weights "$weights" --output "outputs/r1-cohort-$SLURM_JOB_ID"
