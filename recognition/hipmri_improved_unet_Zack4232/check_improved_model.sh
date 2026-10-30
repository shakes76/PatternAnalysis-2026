#!/bin/bash
#SBATCH --job-name=improved-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=improved_check_%j.out
#SBATCH --error=improved_check_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -u check_improved_model.py \
    --baseline-checkpoint runs/ce_dice_711775/results/best.pt
