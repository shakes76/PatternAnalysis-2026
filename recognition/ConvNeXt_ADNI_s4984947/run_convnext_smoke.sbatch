#!/bin/bash
#SBATCH --job-name=adni-convnext
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=convnext-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

echo "=== ConvNeXt GPU Smoke Test ==="
nvidia-smi

python -u train.py \
  --model convnext \
  --epochs 1 \
  --batch-size 4 \
  --slices-per-scan 2 \
  --max-train-batches 10 \
  --output runs/convnext_smoke_v1

echo "=== ConvNeXt Smoke Test Completed ==="
