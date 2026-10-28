#!/bin/bash
#SBATCH --job-name=adni-v3-60
#SBATCH --partition=a100-test
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:08:00
#SBATCH --output=v3long-%j.out
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comp3710-unet
cd "$SLURM_SUBMIT_DIR"
python -u train_v3_long.py --epochs 60 --stage-epochs 10 --batch-size 16 --slices-per-scan 6 --lr 0.0002 --output runs/convnext_v3_60_newsplit
