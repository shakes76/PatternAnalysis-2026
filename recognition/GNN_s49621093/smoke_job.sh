#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=smoke_%j.out

source /home/Student/s4962109/miniconda3/etc/profile.d/conda.sh
conda activate torch

cd ~/PatternAnalysis-2026/recognition/GNN_s49621093
python smoke_test.py