#!/bin/bash
#SBATCH --job-name=gfnet_fail
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%j.out

source ~/.bashrc
conda activate gfnet
cd ~/PatternAnalysis-2026/recognition/GFNet_s4802604
python failure_cases.py --ckpt ~/adni_work/gfnet_clean_r4_stem_s2.pth --conv-stem --preds ~/adni_work/gfnet_clean_r4_stem_s2_test_preds.json --name final
