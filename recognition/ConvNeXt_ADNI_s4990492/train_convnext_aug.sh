#!/bin/bash
#SBATCH --job-name=adni-convnext-aug
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=train_convnext_aug_%j.out
#SBATCH --error=train_convnext_aug_%j.err

set -e
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710_adni_project"

python -u train_convnext_aug.py --model convnext --epochs 20 --batch-size 8
