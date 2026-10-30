#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

# Run the script
python predict.py
