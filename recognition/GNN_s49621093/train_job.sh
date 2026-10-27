#!/bin/bash
#SBATCH --job-name=gnn_train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=gnn_%j.out

source /home/Student/s4962109/miniconda3/etc/profile.d/conda.sh
conda activate torch
cd ~/PatternAnalysis-2026/recognition/GNN_s49621093
for s in 0 1 2; do
    echo "=== seed $s ==="
    python train.py --seed $s --split-seed $s --out outputs/seed$s
done