#!/bin/bash
#SBATCH --job-name=adni-convnext
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=train_convnext_%j.out
#SBATCH --error=train_convnext_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch
set -e

cd "$HOME/comp3710_adni_project"
python -u train.py --model convnext --epochs 3 --batch-size 8