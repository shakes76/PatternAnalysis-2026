#!/bin/bash
#SBATCH --job-name=gfnet_r3
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
BASE="--model gfnet --epochs 30 --lr 3e-4 --dim 128 --depth 4 --wd 0.05 --select scan_acc"
REC="--drop 0.1 --drop-path 0.2 --warmup 3 --cosine --mixup 0.4 --strong-aug --smooth 0.1"
python train.py $BASE $REC --tag _r3_recipe
python train.py $BASE $REC --conv-stem --tag _r3_stem
python train.py $BASE --drop 0.3 --conv-stem --tag _r3_stem_only
