#!/bin/bash
# Separate validation-only diagnostic after run703884 missed labels3,4,5.
# Change training slice sampling only; retain architecture, loss, split and LR.
#SBATCH --job-name=hipmri-balance-pilot
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_balance_%j.out
#SBATCH --error=runs/baseline_balance_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"

# Fresh seed42 model; preserve the failed run's weights/reports for comparison.
python -u train.py --mode fit --device cuda --epochs 3 --batch-size 8 \
    --learning-rate 0.001 --seed 42 --slice-sampling prostate-balanced \
    --manifest runs/split_seed42.json --output "runs/baseline_balance_${SLURM_JOB_ID}"
