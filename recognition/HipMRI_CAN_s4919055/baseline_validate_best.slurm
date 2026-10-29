#!/bin/bash
# Inspect the already trained 709041 best model; no optimization or test data.
#SBATCH --job-name=hipmri-best-val
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_validate_best_%j.out
#SBATCH --error=runs/baseline_validate_best_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"
output_dir="runs/baseline_validate_best_${SLURM_JOB_ID}"
python -u validate_best.py --device cuda \
    --checkpoint runs/baseline_dice_ce_resume_709041/best.pt \
    --expected-sha256 094deac975b46ecd9c0fa497d9581fd188d120bd34c318b9897e6bf3239a303c \
    --output "$output_dir"
tar -czf "${output_dir}_diagnostics.tar.gz" -C "$output_dir" .
echo "BEST_VALIDATION_AND_PANELS_COMPLETED (test cases were not evaluated)."
