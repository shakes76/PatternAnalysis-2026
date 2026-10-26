#!/bin/bash
#SBATCH --job-name=hipmri-cache-identity
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
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
.venv/bin/python -u tools/cache_identity.py record \
  --manifest artifacts/cache64/manifest.json \
  --output artifacts/cache64/cache_identity.json
verify_root="artifacts/rebuild_check_${SLURM_JOB_ID}"
mkdir -p "$verify_root"
cp artifacts/cache64/manifest.json "$verify_root/manifest.json"
# One TRAIN case checks actual NIfTI resampling against the decoded array hashes.
# The supplied manifest bytes remain unchanged; no model is trained or evaluated.
.venv/bin/python -u tools/cache_identity.py rebuild \
  --manifest "$verify_root/manifest.json" \
  --identity artifacts/cache64/cache_identity.json \
  --data-root /home/groups/comp3710/HipMRI_Study_open --split train --limit 1 \
  > "$verify_root/rebuild_result.json"
cat "$verify_root/rebuild_result.json"
