#!/bin/bash
#SBATCH --job-name=wgan-gp-hipmri
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
##SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/wgan_gp_%j.out
#SBATCH --error=slurm/logs/wgan_gp_%j.err

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi

source $HOME/miniconda3/bin/activate
conda activate torch

# Run from the project folder so data/cache and runs/ land here
cd $HOME/PatternAnalysis-2026/recognition/WGAN_GP_HipMRI_s4889661

python train.py \
  --root /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
  --model wgan-gp --epochs ${EPOCHS:-30} \
  --run-name ${RUN_NAME:-wgan_gp}

echo "Finished $(date)"
