#!/bin/bash
# Train the 2D U-Net (baseline) on HipMRI 3D.
# Submit from recognition/ImprovedUNet3D_s49116889:
#   sbatch slurm/train_2d.slurm
# Output log: slurm-train_2d-<jobid>.out (in the submit folder)
#SBATCH --job-name=train_2d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%x-%j.out

source slurm/env.sh
python train.py --model 2d --workers 4
