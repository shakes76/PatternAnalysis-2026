#!/bin/bash
# Extend the three-epoch balanced pilot by five epochs, without changing recipe.
#SBATCH --job-name=hipmri-balance-resume
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_balance_resume_%j.out
#SBATCH --error=runs/baseline_balance_resume_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"

test -f runs/baseline_balance_704488/last.pt
test -f runs/baseline_balance_704488/best.pt
python -u train.py --mode fit --device cuda --epochs 8 --batch-size 8 \
    --learning-rate 0.001 --seed 42 --slice-sampling prostate-balanced \
    --manifest runs/split_seed42.json --resume runs/baseline_balance_704488/last.pt \
    --output "runs/baseline_balance_resume_${SLURM_JOB_ID}"
