#!/bin/bash
#SBATCH --job-name=wgan_train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=wgan_%j.out
#SBATCH --error=wgan_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Run the script
python train.py
