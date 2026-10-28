#!/bin/bash
#SBATCH --job-name=adni-baseline
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=baseline-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

echo "=== Full CNN Baseline Training ==="

python -u train.py \
  --model baseline \
  --epochs 8 \
  --batch-size 32 \
  --slices-per-scan 2 \
  --max-train-batches 0 \
  --output runs/baseline_full_v1

echo "=== Baseline Completed ==="
