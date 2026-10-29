#!/bin/bash
# predict.sh - Slurm job: evaluate a trained model on the test set (one A100, Rangpur).
#
# Author:  Angus Slater (s4961587)
# Course:  COMP3710 Pattern Recognition, The University of Queensland, 2026
# Project: Improved 3D U-Net for prostate MRI segmentation (HipMRI)
#
# Usage:  sbatch predict.sh --model <unet3d|improved> --tag <run tag>
#   e.g.  sbatch predict.sh --model improved --tag v1
#
# Arguments after the script name are passed straight to predict.py.
#SBATCH --job-name=predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err
##SBATCH --mail-user=sXXXXXXX@student.uq.edu.au
##SBATCH --mail-type=END,FAIL

ARGS=("$@")  # save predict.py args, then clear them: conda's activate script would otherwise read them
set --

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi
# Activate a PyTorch environment (see README Section 10.1); edit these two lines for your setup.
source $HOME/miniconda3/bin/activate
conda activate torch
cd "$SLURM_SUBMIT_DIR"   # run from wherever the job was submitted (the project folder)
python -u predict.py "${ARGS[@]}"
echo "finished $(date)"