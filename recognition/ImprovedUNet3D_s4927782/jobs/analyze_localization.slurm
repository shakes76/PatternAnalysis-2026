#!/bin/bash
#SBATCH --job-name=hipmri-localization
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
.venv/bin/python tools/analyze_localization.py \
  --manifest artifacts/cache64/manifest.json \
  --baseline-eval artifacts/test_final/unet2d/evaluation.json \
  --model-eval artifacts/test_final/improved3d/evaluation.json \
  --output artifacts/test_final/localization.json
