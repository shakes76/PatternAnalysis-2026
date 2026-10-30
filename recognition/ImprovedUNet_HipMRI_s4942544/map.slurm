#!/bin/bash
# Link every 2D slice to its source 3D volume, and inherit the 2D patient-level
# split for the 3D model (CPU work only; runs on the course partition).
#
# Usage (from this folder):
#   sbatch map.slurm
#SBATCH --job-name=hipmri-map
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=logs/map_%j.out
#SBATCH --error=logs/map_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
python map_2d_to_3d.py --out mapping_out
echo "Finished $(date)"
