#!/bin/bash
#SBATCH --job-name=adni-phase2-v7
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=runs/phase2-%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet
cd "$SLURM_SUBMIT_DIR"
python -u phase2_make_training_plots.py
python -u phase2_final_test_visuals.py
printf '\n=== PHASE TWO COMPLETE ===\n'
