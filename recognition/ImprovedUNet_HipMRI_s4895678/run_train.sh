#!/bin/bash
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Generic training job. Usage: sbatch --job-name=NAME --time=HH:MM:SS run_train.sh <train.py arguments>
# No --mem line on purpose: this cluster rejects explicit memory requests.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python -u train.py "$@"
