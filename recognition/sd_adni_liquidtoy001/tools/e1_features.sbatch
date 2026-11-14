#!/bin/bash
#SBATCH --job-name=sd-e1-features
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/e1_features_%j.out
#SBATCH --error=logs/e1_features_%j.err
# S3 of docs/e1-protocol.md: frozen-VAE features for the historical cohorts plus
# the E1 cohorts. Submit from the project folder that holds the historical outputs.
# Usage: sbatch tools/e1_features.sbatch --extra e1e10=PATH --extra e1e05=PATH
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
.venv-train/bin/python tools/extract_features.py --output "outputs/features-$SLURM_JOB_ID" "$@"
