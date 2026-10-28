#!/bin/bash
#SBATCH --job-name=adni-first
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

echo "=== GPU information ==="
nvidia-smi

echo "=== Experiment 1: CNN baseline ==="
python -u train.py \
  --model baseline \
  --epochs 2 \
  --batch-size 32 \
  --slices-per-scan 2 \
  --output runs/baseline_day1

echo "=== Experiment 2: ConvNeXt ==="
python -u train.py \
  --model convnext \
  --epochs 1 \
  --batch-size 8 \
  --slices-per-scan 2 \
  --max-train-batches 40 \
  --output runs/convnext_day1

echo "=== Experiments completed ==="
