#!/bin/bash
#SBATCH --job-name=unet-overfit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=overfit_%j.out
#SBATCH --error=overfit_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"

python -u overfit_check.py \
    --steps 400 \
    --output "runs/overfit_${SLURM_JOB_ID}"
