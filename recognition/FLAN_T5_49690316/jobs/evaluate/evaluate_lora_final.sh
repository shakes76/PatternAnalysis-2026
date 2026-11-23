#!/bin/bash
#SBATCH --job-name=flan-lora-final
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=evaluate_lora_final_%j.out
#SBATCH --error=evaluate_lora_final_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$HOME/comp3710/flan_t5"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# Submit only after the corresponding training run completes successfully.
# Pass --resume to this script to reuse saved evaluation progress.
python -u evaluate.py \
    --checkpoint lora_all \
    --split final \
    --samples 10000 \
    --max-new-tokens 1024 \
    --output-name lora_all_final_eval \
    "$@"
