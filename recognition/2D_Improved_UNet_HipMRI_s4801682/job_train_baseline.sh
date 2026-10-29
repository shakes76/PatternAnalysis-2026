#!/bin/bash
#SBATCH --job-name=baseline_unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=train_baseline_%j.out
#SBATCH --error=train_baseline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=s4801682@student.uq.edu.au

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

# Point Linux to Conda's c++ librarys (fixes GLIBCXX error)
export LD_LIBRARY_PATH=$HOME/miniconda3/envs/torch/lib:$LD_LIBRARY_PATH

# Print Execution Metadata
echo "-----------------------"
echo "Running Baseline U-Net Training Test: $(hostname)"
echo "Start Time: $(date)"
echo "-----------------------"

python train.py

echo "-----------------------"
echo "Training script finished at: $(date)"
echo "-----------------------"
