#!/bin/bash
#SBATCH --job-name=adni-v6-slices
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=v6slices-%j.out

set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet
cd "$SLURM_SUBMIT_DIR"

python -u train_v6_12slices.py \
  --epochs 20 \
  --stage-epochs 4 \
  --batch-size 16 \
  --slices-per-scan 12 \
  --lr 0.0002 \
  --output runs/convnext_v6_12slices20_newsplit
