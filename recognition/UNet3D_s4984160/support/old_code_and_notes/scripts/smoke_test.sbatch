#!/bin/bash
#SBATCH --job-name=unet3d_smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err

set -e

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/UNet3D_s4984160"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

nvidia-smi
python -u somke_test.py
