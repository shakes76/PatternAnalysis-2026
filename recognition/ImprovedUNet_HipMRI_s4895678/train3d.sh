#!/bin/bash
#SBATCH --job-name=improved3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:30:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# First 3D Improved U-Net run: lr 5e-4, 50 epochs (the matched run is train3d_matched.sh).

# No --mem line on purpose: this cluster rejects explicit memory requests.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python train.py --model improved3d --base3d 16 --epochs 50 --lr 5e-4
