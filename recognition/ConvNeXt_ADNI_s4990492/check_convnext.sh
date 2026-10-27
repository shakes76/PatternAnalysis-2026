#!/bin/bash
#SBATCH --job-name=adni-convnext-check
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=check_convnext_%j.out
#SBATCH --error=check_convnext_%j.err

source "$HOME/miniconda3/bin/activate"
conda activate torch
set -e

cd "$HOME/comp3710_adni_project"
python -u check_convnext.py