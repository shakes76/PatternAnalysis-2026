#!/bin/bash
#SBATCH --job-name=adni_convnext
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=convnext_%j.out
#SBATCH --error=convnext_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

cd ~/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_YangjiaYan

python -u train.py \
    --model convnext \
    --pretrained \
    --epochs 15 \
    --batch-size 32 \
    --workers 4 \
    --lr 0.0001 \
    --seed 42 \
    --output-dir ~/data/comp3710_a3_runs_pretrained
