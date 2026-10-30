#!/bin/bash
#SBATCH --job-name=unet-inspect
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=inspect_%j.out
#SBATCH --error=inspect_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python -u predict.py \
    --checkpoint runs/baseline_710034/results/best.pt \
    --output "runs/validation_inspection_${SLURM_JOB_ID}"
