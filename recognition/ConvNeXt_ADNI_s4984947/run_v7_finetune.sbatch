#!/bin/bash
#SBATCH --job-name=adni-v7-pre
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=v7pre-%j.out

set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet
cd "$SLURM_SUBMIT_DIR"

python -u train_v7_finetune.py \
  --epochs 20 \
  --stage-epochs 4 \
  --batch-size 16 \
  --slices-per-scan 6 \
  --head-warmup-epochs 2 \
  --body-lr 0.00002 \
  --head-lr 0.0003 \
  --weights pretrained/convnext_tiny-983f1562.pth \
  --output runs/convnext_v7_pretrained20_newsplit
