#!/bin/bash
#SBATCH --job-name=unet-reg
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=reg_%j.out
#SBATCH --error=reg_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$HOME/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI" || exit 1

python -u train.py \
    --model improved \
    --loss ce \
    --epochs 10 \
    --batch-size 4 \
    --lr 0.0001 \
    --weight-decay 0.0001 \
    --seed 42 \
    --run-name improved_reg_v1
