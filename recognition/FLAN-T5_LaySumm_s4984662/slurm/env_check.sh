#!/bin/bash
# env_check.sh - Slurm job: check that the GPU environment works (see env_check.py).
# Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662) | COMP3710 Project 2.6
# Usage  : sbatch slurm/env_check.sh      (from the project folder)
#SBATCH --job-name=env-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=env_check_%j.out
#SBATCH --error=env_check_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
free -h

# A batch job starts from a clean environment, so activate conda here.
# Passing "torch" here keeps activate from reading this script's own arguments.
source "$HOME/miniconda3/bin/activate" torch
export HF_HOME="$HOME/hf_cache" HF_HUB_OFFLINE=1   # model and data are pre-downloaded

python slurm/env_check.py
