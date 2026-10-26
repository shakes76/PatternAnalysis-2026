#!/bin/bash
#SBATCH --job-name=hipmri-seed-followup-analysis
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# Submit once, afterok of the one paired seed43 job. No GPU or TEST evaluation.
.venv/bin/python -m unittest discover -s tests -p test_seed_followup.py -v
.venv/bin/python tools/analyze_seed_followup.py \
  --seed42-root artifacts/full_seed42 \
  --seed43-root artifacts/followup_seed43 \
  --seed42-val artifacts/validation_seed42 \
  --seed43-val artifacts/validation_seed43 \
  --manifest artifacts/cache64/manifest.json \
  --protocol followup_protocol.json \
  --output-dir "artifacts/seed_followup_summary_${SLURM_JOB_ID}"
