#!/bin/bash
#SBATCH --job-name=failcs_unet
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=failcs_%j.out
#SBATCH --error=failcs_%j.err

cd ~/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI

source "$HOME/miniconda3/bin/activate"
conda activate torch

python failcs.py

