#!/bin/bash
#SBATCH --job-name=hipmri-failure-details
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
.venv/bin/python -u tools/plot_failure_details.py \
  --manifest artifacts/cache64/manifest.json \
  --baseline-eval artifacts/test_final/unet2d/evaluation.json \
  --model-eval artifacts/test_final/improved3d/evaluation.json \
  --pair-analysis artifacts/test_final/paired/pair_analysis.json \
  --output-dir artifacts/test_final/details
