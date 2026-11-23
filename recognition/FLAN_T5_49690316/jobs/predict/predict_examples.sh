#!/bin/bash
#SBATCH --job-name=flan-predict-demo
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=predict_examples_%j.out
#SBATCH --error=predict_examples_%j.err

set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/comp3710/flan_t5"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Exercise the final prediction interface; no training or additional scoring.
python -u predict.py --checkpoint lora_all --split final --samples 3 \
    --device cuda --output "results/predict_lora_${SLURM_JOB_ID}.json"
python -u predict.py --checkpoint full_all --split final --samples 3 \
    --device cuda --output "results/predict_full_${SLURM_JOB_ID}.json"
