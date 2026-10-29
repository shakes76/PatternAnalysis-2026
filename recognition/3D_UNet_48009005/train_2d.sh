#!/bin/bash
#SBATCH --job-name=HIPMRI_2D_UNET
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:50:00

#SBATCH --output=2D_UNET_%j.out
#SBATCH --error=2D_UNET_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

cd $SLURM_SUBMIT_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

nvidia-smi

python -u train.py


echo "Finished: $(date)"
