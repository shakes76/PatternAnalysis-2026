#!/bin/bash
#SBATCH --job-name=adni-convnext-v1
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=convnext-v1-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

echo "=== ConvNeXt Full-Subset Training ==="

python -u train.py \
    --model convnext \
    --epochs 2 \
    --batch-size 8 \
    --slices-per-scan 2 \
    --max-train-batches 0 \
    --lr 0.0001 \
    --output runs/convnext_v1

echo "=== ConvNeXt Training Completed ==="
