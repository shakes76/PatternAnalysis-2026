#!/bin/bash
#SBATCH --job-name=gfnet_r5
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
REC="--epochs 30 --lr 3e-4 --wd 0.05 --drop 0.1 --drop-path 0.2 --warmup 3 --cosine --mixup 0.4 --strong-aug --smooth 0.1 --select scan_acc"
for s in 42 1 2; do
  python train.py --model gfnet --dim 128 --depth 4 $REC --seed $s --tag _r5_plain_s$s
  python train.py --model gfnet --dim 128 --depth 0 --conv-stem $REC --seed $s --tag _r5_nogf_s$s
done
python train.py --model gfnet --dim 128 --depth 4 --conv-stem $REC --seed 42 --tag _r5_stem_s42
