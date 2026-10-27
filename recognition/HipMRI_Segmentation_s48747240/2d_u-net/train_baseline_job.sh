#!/bin/bash
#SBATCH --job-name=hipmri-baseline-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --exclude=a100-2
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate tf

# Run training
cd $HOME/PatternAnalysis-2026/recognition/HipMRI_Segmentation_s48747240
python -u train.py baseline
