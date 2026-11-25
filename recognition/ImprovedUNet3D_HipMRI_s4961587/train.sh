#!/bin/bash
# train.sh - Slurm job: train a model on one A100 on Rangpur.
#
# Author:  Angus Slater (s4961587)
# Course:  COMP3710 Pattern Recognition, The University of Queensland, 2026
# Project: Improved 3D U-Net for prostate MRI segmentation (HipMRI)
#
# Usage:  sbatch train.sh [train.py args]
   #   smoke:     sbatch --time=00:20:00 train.sh --model improved --epochs 3 --limit 4 --tag smoke
   #   baseline:  sbatch train.sh --model unet3d --epochs 150 --tag baseline
   #   improved:  sbatch train.sh --model improved --epochs 150 --tag v1
#
# Arguments after the script name are passed straight to train.py.
#SBATCH --job-name=unet3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=unet3d_%j.out
#SBATCH --error=unet3d_%j.err
##SBATCH --mail-user=sXXXXXXX@student.uq.edu.au
##SBATCH --mail-type=END,FAIL

ARGS=("$@")  # save train.py args, then clear them: conda's activate script would otherwise read them
set --

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
# Activate a PyTorch environment (see README Section 10.1); edit these two lines for your setup.
source $HOME/miniconda3/bin/activate
conda activate torch
cd "$SLURM_SUBMIT_DIR"   # run from wherever the job was submitted (the project folder)
python -u train.py "${ARGS[@]}"
echo "finished $(date)"