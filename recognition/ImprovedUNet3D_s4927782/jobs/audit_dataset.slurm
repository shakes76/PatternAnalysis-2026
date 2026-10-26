#!/bin/bash
#SBATCH --job-name=hipmri-data-audit
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
mkdir -p "$TMPDIR"
.venv/bin/python -m unittest discover -s tests -p 'test_audit*.py' -v
.venv/bin/python tools/audit_dataset.py \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --inspect-labels-limit 3 --output artifacts/dataset_audit.json
