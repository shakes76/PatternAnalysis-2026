#!/bin/bash
#SBATCH --job-name=adni-baseline
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=train_baseline_%j.out
#SBATCH --error=train_baseline_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch
set -e

cd "$HOME/comp3710_adni_project"
python -u train.py --epochs 3 --batch-size 8