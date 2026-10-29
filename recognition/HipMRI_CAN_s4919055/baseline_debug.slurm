#!/bin/bash
# Course Ed #75 GPU setup: one short training-only baseline debugging run.
#SBATCH --job-name=hipmri-baseline-debug
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --output=runs/baseline_debug_%j.out
#SBATCH --error=runs/baseline_debug_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/runs/mpl-cache"
python -u train.py --mode debug --device cuda --debug-steps 100 --output "runs/baseline_debug_${SLURM_JOB_ID}"
python -u predict.py --checkpoint "runs/baseline_debug_${SLURM_JOB_ID}/debug.pt" --split train --device cuda --output "runs/baseline_preview_${SLURM_JOB_ID}"
