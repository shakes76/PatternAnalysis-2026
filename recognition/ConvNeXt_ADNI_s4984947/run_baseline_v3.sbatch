#!/bin/bash
#SBATCH --job-name=adni-baseline-v3
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=baseline-v3-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

echo "=== CNN Baseline V3: New Patient Split ==="

python -u train_v3_baseline.py \
    --epochs 15 \
    --batch-size 16 \
    --slices-per-scan 6 \
    --lr 0.0002 \
    --output runs/baseline_v3_15epochs

echo "=== Baseline V3 Completed ==="
