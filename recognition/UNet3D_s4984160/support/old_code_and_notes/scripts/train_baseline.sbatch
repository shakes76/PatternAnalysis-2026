#!/bin/bash
#SBATCH --job-name=unet3d_baseline
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:15:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

set -e

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/UNet3D_s4984160"

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

python -u train.py \
    --epochs 1 \
    --max-train-batches 2 \
    --max-val-batches 2 \
    --output-dir results/baseline_pilot
