#!/bin/bash
#SBATCH --job-name=gfnet_r2
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
COMMON="--model gfnet --epochs 20 --lr 3e-4 --dim 64 --depth 2 --drop 0.3 --wd 0.05 --select scan_acc"
python train.py $COMMON --tag _r2_sel
python train.py $COMMON --strong-aug --smooth 0.1 --tag _r2_aug
python train.py $COMMON --strong-aug --smooth 0.1 --patch 8 --tag _r2_p8
