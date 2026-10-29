#!/bin/bash
#SBATCH --job-name=timegan
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1

# Trains the model chosen by MODEL in train.py on one GPU.
# Submit with: sbatch --export=ALL,MODEL=timegan,SMOKE_TEST=0 run_train.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/comp3710_lab2/pytorch-env
python train.py