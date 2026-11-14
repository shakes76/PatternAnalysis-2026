#!/bin/bash
#SBATCH --job-name=sd-neighbors
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=logs/neighbors_%j.out
#SBATCH --error=logs/neighbors_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
root="outputs/cohort-$1"
.venv-train/bin/python tools/audit_cohort.py "$root/base" "$root/adapted" --output "$root/audit"
