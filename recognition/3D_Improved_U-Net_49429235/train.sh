#!/bin/bash
#SBATCH --job-name=improved3d
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/cpu_smoke_%j.out
#SBATCH --error=logs/cpu_smoke_%j.err


source $HOME/miniconda3/bin/activate
conda activate torch
cd ~/PatternAnalysis-2026/recognition

CASES="J026_Week1 T014_Week2 T014_Week1 J026_Week7 J026_Week6"

python predict.py --checkpoint runs/improved/best.pt \
    --split-file runs/improved/splits.json \
    --out-dir predictions/improved --cases $CASES

python predict.py --checkpoint runs/baseline/best.pt \
    --split-file runs/improved/splits.json \
    --out-dir predictions/baseline --cases $CASES