#!/bin/bash
# train.sh - Slurm job: train and evaluate one configuration (zero-shot, LoRA or full FT).
# Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662) | COMP3710 Project 2.6
# Usage  : sbatch slurm/train.sh --mode lora --lora_r 8 --lr 3e-4    (from the project folder)
#          Every argument is passed to train.py; see `python train.py --help`.
#SBATCH --job-name=rrs-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=run_%j.out
#SBATCH --error=run_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# A batch job starts from a clean environment, so activate conda and set the cache here.
# Passing "torch" here keeps activate from reading this script's own arguments.
source "$HOME/miniconda3/bin/activate" torch
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
echo "code version: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "arguments: $@"

# One process per GPU. The rendezvous stays on this node: loopback address and a port
# derived from the job id.
torchrun --nnodes=1 --nproc_per_node=1 --local_addr=127.0.0.1 \
         --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:$((20000 + SLURM_JOB_ID % 20000)) \
         train.py "$@"
