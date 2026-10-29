#!/bin/bash
#SBATCH --job-name=3D_TRAIN_PLOT
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --cpus-per-task=2
#SBATCH --time=00:05:00
#SBATCH --output=3D_PLOT_%j.out
#SBATCH --error=3D_PLOT_%j.err

set -e

source $HOME/miniconda3/bin/activate
conda activate torch

cd $SLURM_SUBMIT_DIR

python -u -c "from train import plot_3d_training; plot_3d_training()"
