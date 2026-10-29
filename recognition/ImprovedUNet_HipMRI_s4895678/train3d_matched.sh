#!/bin/bash
#SBATCH --job-name=i3d_matched
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:30:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err

# Matched control: same learning rate and epochs as the 2D baseline.
# No --mem line on purpose: this cluster rejects explicit memory requests.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python -u train.py --model improved3d --base3d 16 --epochs 40 --lr 1e-3 --tag matched
