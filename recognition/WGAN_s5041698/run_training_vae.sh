#!/bin/bash
#SBATCH --job-name=vae_train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=vae_%j.out
#SBATCH --error=vae_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Run the script
python train_vae.py
