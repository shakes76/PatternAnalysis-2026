#!/bin/bash
#SBATCH --job-name=predict_unet
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

cd ~/PatternAnalysis-2026/recognition/Improved_UNet_HipMRI

source "$HOME/miniconda3/bin/activate"
conda activate torch

python predict.py --indices 267 207 513 453 --output-dir evaluation/failure_cases
