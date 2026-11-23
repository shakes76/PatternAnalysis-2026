#!/bin/bash
#SBATCH --job-name=flan-lora-pilot
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=train_pilot_%j.out
#SBATCH --error=train_pilot_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$HOME/comp3710/flan_t5"

export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1
# Tells huggingface to use cached files without network equests. 
# Missing cached files will produce an error rather than spend GPU time downloading

echo "Job $SLURM_JOB_ID on $(hostname)"
# Logs the job iD and compute node name. $(hostname) inserts the output of the hostname command
nvidia-smi

torchrun --standalone --nnodes=1 --nproc_per_node=1 train.py \
    --mode lora \
    --learning-rate 0.001 \
    --run-name pilot_lora_rangpur