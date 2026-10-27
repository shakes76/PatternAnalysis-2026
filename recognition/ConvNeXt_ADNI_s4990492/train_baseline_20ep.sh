#!/bin/bash
#SBATCH --job-name=adni-baseline20
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=train_baseline_20ep_%j.out
#SBATCH --error=train_baseline_20ep_%j.err

set -e
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710_adni_project"

python -u train.py --model baseline --epochs 20 --batch-size 8