#!/bin/bash
#SBATCH --job-name=hipmri-train
#SBATCH --account=comp3710
#SBATCH --partition=comp3710,a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# --account=comp3710 is required: the comp3710 partition rejects the default
# personal account. No --mem: Rangpur nodes report RealMemory=1, so any --mem
# request pends forever.

# Train one model on an A100. Arguments go straight to train.py:
#     sbatch -J unet3d slurm/train.sh --model unet3d
#     sbatch -J unet2d slurm/train.sh --model unet2d

set -euo pipefail
echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1   # stream prints to the log as they happen
python train.py "$@"
echo "done $(date)"
