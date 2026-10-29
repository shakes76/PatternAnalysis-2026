#!/bin/bash
#SBATCH --job-name=hipmri-unets
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=8:00:00
#SBATCH --output=hipmri_%j.out
#SBATCH --error=hipmri_%j.err

# Exit if any command fails.
set -euo pipefail

# Run from the folder where sbatch was submitted (contains train.py).
cd "$SLURM_SUBMIT_DIR"

# Activate the same Conda environment as the working CUDA test.
source "$HOME/miniconda3/bin/activate"
conda activate torch

echo "Job ID: $SLURM_JOB_ID"
echo "Project directory: $PWD"
echo "Python: $(command -v python)"
echo "Started: $(date)"
nvidia-smi

# Make sure PyTorch has GPU access and the shared HipMRI data is present.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print("GPU:", torch.cuda.get_device_name(0))'
test -d /home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices_train

# Your train.py trains baseline first, then improved, saving them separately.
python -u train.py

echo "Finished: $(date)"
