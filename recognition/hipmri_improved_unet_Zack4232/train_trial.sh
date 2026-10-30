#!/bin/bash
#SBATCH --job-name=unet-trial
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=train_trial_%j.out
#SBATCH --error=train_trial_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

python -u train.py \
    --epochs 3 \
    --batch-size 8 \
    --output "runs/train_trial_${SLURM_JOB_ID}"
