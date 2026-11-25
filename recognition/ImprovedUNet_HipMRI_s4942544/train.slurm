#!/bin/bash
# Full training run for one model and one random seed.
#
# Usage (from this folder):
#   sbatch --export=ALL,MODEL=baseline,SEED=0 train.slurm
#   sbatch --export=ALL,MODEL=improved,SEED=1 train.slurm
# Optional: EPOCHS=80 (default 50); EXTRA="..." passes further train.py options.
# The 3D Improved U-Net uses train3d.slurm instead.
#
# Note: --account=comp3710 is required -- the comp3710 partition only accepts jobs
# charged to the comp3710 account, and otherwise leaves them pending with reason
# PartitionConfig. Without --cpus-per-task the job gets one CPU core, which two
# data-loader workers suit.
#SBATCH --job-name=hipmri-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:15:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

MODEL=${MODEL:-improved}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-50}
EXTRA=${EXTRA:-}

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date) | model=$MODEL seed=$SEED epochs=$EPOCHS extra=$EXTRA"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py --model "$MODEL" --seed "$SEED" --epochs "$EPOCHS" --workers 2 --out "runs/${MODEL}_s${SEED}" $EXTRA

echo "Finished $(date)"
