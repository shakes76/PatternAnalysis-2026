#!/bin/bash
#SBATCH --job-name=flan-lora-dev
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=train_development_%j.out
#SBATCH --error=train_development_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$HOME/comp3710/flan_t5"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --mode lora \
    --learning-rate 0.001 \
    --train-size 10000 \
    --dev-size 1000 \
    --batch-size 2 \
    --epochs 3 \
    --save-every 500 \
    --run-name lora_dev_10k