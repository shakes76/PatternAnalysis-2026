#!/bin/bash
#SBATCH --job-name=gfnet_pred
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
for ck in r4_stem_s1 r4_stem_s2 r5_stem_s42; do
  for sp in val test; do
    python predict.py --ckpt ~/adni_work/gfnet_clean_$ck.pth --model gfnet --conv-stem --split $sp
  done
done
for sp in val test; do
  python predict.py --ckpt ~/adni_work/baseline_clean_r4_cnn_s42.pth --model baseline --split $sp
done
