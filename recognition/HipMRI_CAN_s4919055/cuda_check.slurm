#!/bin/bash
# Course Ed #6/#75 updated GPU allocation pattern, checked 15 October 2026.
#SBATCH --job-name=hipmri-cuda-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=runs/cuda_check_%j.out
#SBATCH --error=runs/cuda_check_%j.err

set -e
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch
python -u cuda_check.py
