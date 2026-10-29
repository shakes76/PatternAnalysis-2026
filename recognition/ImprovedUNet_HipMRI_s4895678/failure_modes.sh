#!/bin/bash
#SBATCH --job-name=failmodes
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:20:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Count dropout and over-segmented slices for the three models (failure_modes.py).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python -u failure_modes.py
