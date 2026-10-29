#!/bin/bash
#SBATCH --job-name=smoke3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Quick 3D end-to-end check (1 epoch, 4 volumes per split). Writes *_smoke files, so real results are not overwritten.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python train.py --model improved3d --base3d 16 --epochs 1 --max_cases 4 --tag smoke
