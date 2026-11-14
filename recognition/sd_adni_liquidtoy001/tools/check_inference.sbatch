#!/bin/bash
#SBATCH --job-name=sd-infer-audit
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/check_inference_%j.out
#SBATCH --error=logs/check_inference_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-train/bin/python tools/check_inference.py \
    --checkpoint ../runs/full20k-20261004/outputs/full-629931/step-20000/checkpoint.pt \
    --output "outputs/inference-$SLURM_JOB_ID" "$@"
