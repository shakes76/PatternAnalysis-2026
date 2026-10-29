#!/bin/bash
# Fixed validation-selected checkpoints; no training or test-driven tuning.
#SBATCH --job-name=hipmri-final-compare
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/final_compare_%j.out
#SBATCH --error=runs/final_compare_%j.err

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"
output_dir="runs/final_compare_${SLURM_JOB_ID}"
# Separate processes ensure one model's cached allocations do not affect the other.
python -u final_evaluate.py --model can3d --device cuda --final-test \
    --plan final_evaluation_plan.json --output "$output_dir/can3d"
python -u final_evaluate.py --model unet2d --device cuda --final-test \
    --plan final_evaluation_plan.json --output "$output_dir/unet2d" \
    --comparison-results "$output_dir/can3d/test_metrics.json"
tar -czf "${output_dir}_diagnostics.tar.gz" -C "$output_dir" can3d unet2d
echo "FINAL_COMPARISON_COMPLETED (fixed checkpoints; no training updates)."
