#!/bin/bash
#SBATCH --job-name=hipmri-predict
#SBATCH --account=comp3710
#SBATCH --partition=comp3710,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# See train.sh for why --account is set and --mem is not.

# Evaluate trained models on the test patients and draw the README figures:
#     sbatch slurm/predict.sh --checkpoints runs/unet3d/best.pt runs/unet2d/best.pt

set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1   # stream prints to the log as they happen
python predict.py "$@"
echo "done $(date)"
