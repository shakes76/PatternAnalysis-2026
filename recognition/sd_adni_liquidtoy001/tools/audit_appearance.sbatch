#!/bin/bash
#SBATCH --job-name=sd-appearance
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=logs/appearance_%j.out
#SBATCH --error=logs/appearance_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
.venv-train/bin/python tools/audit_appearance.py --real --baseline ../runs/baseline128-20261005/outputs/baseline-vae-632570 --output outputs/appearance-real-20261007
