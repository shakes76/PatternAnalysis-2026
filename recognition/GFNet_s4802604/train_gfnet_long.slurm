#!/bin/bash
#SBATCH --job-name=gfnet_long
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
python train.py --model gfnet --epochs 40 --lr 3e-4 --tag _long
