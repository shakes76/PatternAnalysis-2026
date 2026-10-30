#!/bin/bash
#SBATCH --job-name=flan-inference-profile
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=profile_inference_%j.out
#SBATCH --error=profile_inference_%j.err

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch
cd "$HOME/comp3710/flan_t5"

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi

# Separate Python processes prevent previous models from occupying GPU memory.
python -u -m scripts.analysis.profile_inference --output-name baseline
python -u -m scripts.analysis.profile_inference --checkpoint lora_all --output-name lora_all
python -u -m scripts.analysis.profile_inference --checkpoint full_all --output-name full_all
