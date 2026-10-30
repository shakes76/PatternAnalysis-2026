#!/bin/bash
#SBATCH --job-name=profile_unet
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=profile_%j.out
#SBATCH --error=profile_%j.err

cd ~/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI

source "$HOME/miniconda3/bin/activate"
conda activate torch

python profile_models.py
