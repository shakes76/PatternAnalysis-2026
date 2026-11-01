#!/bin/bash
#SBATCH --job-name=train-unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

MODEL=$1
echo "Job $SLURM_JOB_ID: $MODEL on $(hostname), started $(date)"
nvidia-smi
source $HOME/miniconda3/bin/activate
conda activate torch
cd $HOME/improved_unet
python train.py --root /home/groups/comp3710/OASIS --model $MODEL --base-channels 16 --epochs 15 --num-workers 4
