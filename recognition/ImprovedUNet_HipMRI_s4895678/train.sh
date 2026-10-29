#!/bin/bash
#SBATCH --job-name=unet2d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Baseline 2D U-Net, 40 epochs, seed 0.

# Usage: sbatch train.sh        (logs go to outputs/, which git ignores)
# No --mem line on purpose: this cluster rejects explicit memory requests.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python train.py --model unet2d --epochs 40
