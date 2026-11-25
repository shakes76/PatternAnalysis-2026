#!/bin/bash
# Training run for the 3D Improved U-Net (one random seed, one resolution).
#
# Needs mapping_out/ from map_2d_to_3d.py (the 3D patient-level split).
# Usage (from this folder):
#   sbatch --export=ALL,SEED=0,DOWNSAMPLE=1 train3d.slurm    # full resolution, 256x128x128 input
#   sbatch --export=ALL,SEED=0,DOWNSAMPLE=2 train3d.slurm    # half resolution, 128x64x64 input
# Optional: EPOCHS (default 100), BATCH (default 2), MINUTES (training budget, default 200).
#
# The training budget is kept below the Slurm limit so that, if the epochs do not
# all fit, train.py stops early and still evaluates the best checkpoint.
# Memory: the default allocation is used (as for the 2D jobs). Requesting --mem=32G
# made the job unschedulable on this partition (reason BadConstraints). The 3D
# training set needs ~2.5 GB of RAM (float16 images + uint8 masks).
#SBATCH --job-name=hipmri-3d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train3d_%j.out
#SBATCH --error=logs/train3d_%j.err

SEED=${SEED:-0}
DOWNSAMPLE=${DOWNSAMPLE:-1}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-2}
MINUTES=${MINUTES:-200}
if [ "$DOWNSAMPLE" = "1" ]; then RUN="improved3d_s${SEED}"; else RUN="improved3d-ds${DOWNSAMPLE}_s${SEED}"; fi

source $HOME/miniconda3/bin/activate
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date) | run=$RUN epochs=$EPOCHS batch=$BATCH budget=${MINUTES}min"
nvidia-smi --query-gpu=name,memory.total --format=csv

python train.py --model improved3d --seed "$SEED" --epochs "$EPOCHS" --batch-size "$BATCH" \
    --downsample "$DOWNSAMPLE" --max-minutes "$MINUTES" --workers 2 --out "runs/$RUN"

echo "Finished $(date)"
