#!/bin/bash
#SBATCH --job-name=adni-quick
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=quick-%j.out

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet

cd "$SLURM_SUBMIT_DIR"

python -u train.py \
  --model baseline \
  --epochs 1 \
  --batch-size 16 \
  --slices-per-scan 2 \
  --max-train-batches 20 \
  --output runs/baseline_quick
