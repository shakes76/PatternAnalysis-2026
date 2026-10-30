#!/bin/bash
#SBATCH --job-name=unet-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --output=unet_check_%j.out
#SBATCH --error=unet_check_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"

python -u check_model.py
