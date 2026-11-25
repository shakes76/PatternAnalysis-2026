#!/bin/bash
#SBATCH --job-name=unet-eval
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=eval_%j.out
#SBATCH --error=eval_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI" || exit 1

python -u evaluate.py --batch-size 4
