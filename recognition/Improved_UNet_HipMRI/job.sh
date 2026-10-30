#!/bin/bash
#SBATCH --job-name=unet-baseline
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI" || exit 1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

python -u train.py \
    --model baseline \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.0001 \
    --run-name baseline_v1

