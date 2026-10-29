#!/bin/bash
#SBATCH --job-name=HIPMRI_3D_TRAIN_TEST
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

#SBATCH --output=3D_TRAIN_TEST_%j.out
#SBATCH --error=3D_TRAIN_TEST_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

cd $SLURM_SUBMIT_DIR

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

nvidia-smi

python -u -c "from train import train_3d; train_3d(epochs=1, batch_size=1)"

echo "Finished: $(date)"
