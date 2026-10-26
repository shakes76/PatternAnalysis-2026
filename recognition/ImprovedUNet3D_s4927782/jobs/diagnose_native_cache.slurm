#!/bin/bash
#SBATCH --job-name=hipmri-native-diagnosis
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/verify_frozen_evidence.py \
  --manifest artifacts/cache64/manifest.json --protocol test_protocol.json \
  --evaluation artifacts/validation_seed42/unet2d/evaluation.json \
  --evaluation artifacts/validation_seed42/improved3d/evaluation.json \
  --evaluation artifacts/test_final/unet2d/evaluation.json \
  --evaluation artifacts/test_final/improved3d/evaluation.json \
  --output "artifacts/diagnostic_saved_${SLURM_JOB_ID}.json"
.venv/bin/python tools/diagnose_native_cache.py \
  --manifest artifacts/cache64/manifest.json \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --output-dir "artifacts/diagnostic_native_${SLURM_JOB_ID}"
