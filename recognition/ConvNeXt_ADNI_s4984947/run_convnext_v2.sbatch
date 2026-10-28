#!/bin/bash
#SBATCH --job-name=adni-convnext-v2
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=convnext-v2-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

echo "=== ConvNeXt V2 Training ==="

python -u train_v2.py \
    --epochs 8 \
    --batch-size 16 \
    --slices-per-scan 6 \
    --lr 0.0002 \
    --output runs/convnext_v2

echo "=== ConvNeXt V2 Completed ==="
