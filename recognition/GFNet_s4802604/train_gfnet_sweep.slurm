#!/bin/bash
#SBATCH --job-name=gfnet_sweep
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
python train.py --model gfnet --epochs 20 --lr 3e-4 --dim 64  --depth 2 --drop 0.3 --wd 0.05 --tag _d64_l2
python train.py --model gfnet --epochs 20 --lr 3e-4 --dim 128 --depth 4 --drop 0.3 --wd 0.05 --tag _d128_l4_reg
python train.py --model gfnet --epochs 20 --lr 3e-4 --dim 32  --depth 2 --drop 0.3 --wd 0.05 --tag _d32_l2
