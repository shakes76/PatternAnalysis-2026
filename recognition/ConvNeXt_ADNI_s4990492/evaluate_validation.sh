#!/bin/bash
#SBATCH --job-name=adni-validation
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=validation_%j.out
#SBATCH --error=validation_%j.err

set -e
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710_adni_project"

python -u predict.py --checkpoint runs/baseline_20261022_211305_056381/best_baseline.pth
python -u predict.py --checkpoint runs/convnext_aug_lr3e5_20261024_043935_000396/best_convnext.pth
