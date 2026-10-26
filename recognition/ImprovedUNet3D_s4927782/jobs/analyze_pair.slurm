#!/bin/bash
#SBATCH --job-name=hipmri-paired-analysis
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
evidence_dir="${1:-artifacts/test_final}"
.venv/bin/python -u tools/analyze_pair.py --manifest artifacts/cache64/manifest.json \
  --baseline-eval "$evidence_dir/unet2d/evaluation.json" \
  --model-eval "$evidence_dir/improved3d/evaluation.json" \
  --output-dir "$evidence_dir/paired" --target-class 5 \
  --bootstrap-samples 2000 --seed 3710
