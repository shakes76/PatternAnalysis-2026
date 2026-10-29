#!/bin/bash
# Course Ed #75 allocation settings; initial 2D baseline fit, no test evaluation.
#SBATCH --job-name=hipmri-baseline-fit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_fit_%j.out
#SBATCH --error=runs/baseline_fit_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"

# Start a fresh fit on all training cases; select best.pt using validation only.
# If the time limit interrupts training, preserve last.pt and best.pt for resume.
python -u train.py --mode fit --device cuda --epochs 30 --batch-size 8 \
    --learning-rate 0.001 --seed 42 --manifest runs/split_seed42.json \
    --output "runs/baseline_fit_${SLURM_JOB_ID}"
