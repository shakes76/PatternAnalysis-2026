#!/bin/bash
# Test-set inference + README figures for both trained models.
# Submit from recognition/ImprovedUNet3D_s49116889:
#   sbatch slurm/predict.slurm
#SBATCH --job-name=predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out

source slurm/env.sh
python train.py --model 3d --plot-only
python train.py --model 2d --plot-only
python predict.py --models 3d 2d
