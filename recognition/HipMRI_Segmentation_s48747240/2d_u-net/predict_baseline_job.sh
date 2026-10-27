#!/bin/bash
#SBATCH --job-name=hipmri-baseline-predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --exclude=a100-2
#SBATCH --time=00:20:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate tf

# Run inference on the trained model
cd $HOME/PatternAnalysis-2026/recognition/HipMRI_Segmentation_s48747240
python -u predict.py baseline
