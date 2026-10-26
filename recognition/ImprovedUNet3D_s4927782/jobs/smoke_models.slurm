#!/bin/bash
#SBATCH --job-name=hipmri-model-smoke
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
mkdir -p "$TMPDIR"
# Use the scheduler's configured memory default; see AUDIT_SUMMARY.md.
.venv/bin/python -m unittest discover -s tests -p 'test_models.py' -v
.venv/bin/python tools/smoke_models.py \
  --device cuda --base-channels 4 --shape3d 16 32 32 \
  --output artifacts/model_smoke_small.json
.venv/bin/python tools/smoke_models.py \
  --device cuda --base-channels 8 --shape3d 64 128 128 \
  --output artifacts/model_smoke_patch.json
