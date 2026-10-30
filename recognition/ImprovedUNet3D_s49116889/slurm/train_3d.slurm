#!/bin/bash
# Train the 3D Improved U-Net (proposed) on HipMRI 3D.
# Submit from recognition/ImprovedUNet3D_s49116889:
#   sbatch slurm/train_3d.slurm
# Output log: slurm-train_3d-<jobid>.out (in the submit folder)
#SBATCH --job-name=train_3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=slurm-%x-%j.out

source slurm/env.sh
python train.py --model 3d --workers 4
