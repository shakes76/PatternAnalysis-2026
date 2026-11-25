#!/bin/bash
# Feasibility smoke test: 5 epochs of each model on the full HipMRI 2D data.
# Purpose: measure real per-epoch time, peak GPU memory and early Dice for the
# feasibility review -- NOT to produce final results.
#
# Submit from this folder:  sbatch smoke_test.slurm
#SBATCH --job-name=hipmri-smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=smoke_%j.out
#SBATCH --error=smoke_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py --model baseline --epochs 5 --out runs/smoke_baseline
python train.py --model improved --epochs 5 --out runs/smoke_improved

echo "Finished $(date)"
