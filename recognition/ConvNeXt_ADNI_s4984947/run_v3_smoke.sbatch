#!/bin/bash
#SBATCH --job-name=adni-v3-smoke
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=v3-smoke-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet
cd "$SLURM_SUBMIT_DIR"

echo "=== NEW SPLIT: CNN BASELINE ==="

python -u train_v3_baseline.py \
  --epochs 1 \
  --batch-size 16 \
  --slices-per-scan 6 \
  --lr 0.0002 \
  --output runs/v3_baseline_smoke

echo "=== NEW SPLIT: CONVNEXT ==="

python -u train_v3.py \
  --epochs 1 \
  --batch-size 16 \
  --slices-per-scan 6 \
  --lr 0.0002 \
  --output runs/v3_convnext_smoke

echo "=== V3 SMOKE TEST COMPLETED ==="
