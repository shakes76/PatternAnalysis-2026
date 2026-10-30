#!/bin/bash
#SBATCH --job-name=flan-full-eval
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=evaluate_full_development_%j.out
#SBATCH --error=evaluate_full_development_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$HOME/comp3710/flan_t5"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

python -u evaluate.py \
    --checkpoint full_dev_10k \
    --samples 1000 \
    --max-new-tokens 1024 \
    --output-name full_dev_10k_eval