#!/bin/bash
#SBATCH --job-name=flan-full-all
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=train_full_all_%j.out
#SBATCH --error=train_full_all_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$HOME/comp3710/flan_t5"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --mode full \
    --learning-rate 0.0001 \
    --train-size 142801 \
    --dev-size 7515 \
    --batch-size 2 \
    --epochs 3 \
    --save-every 500 \
    --run-name full_all
