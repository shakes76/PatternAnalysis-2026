#!/bin/bash
# Slurm job for training on the Rangpur cluster (COMP3710 partition).
#
# Submit from this folder. Settings come from environment variables:
#   sbatch job_train.sh                                         # full run, CAN
#   MODEL=unet sbatch job_train.sh                              # full run, baseline
#   MODEL=can EPOCHS=2 LIMIT=500 sbatch --time=00:20:00 job_train.sh        # quick test
#   MODEL=can SEED=1 RUN_NAME=can_s1 sbatch job_train.sh                  # repeat with another seed
#   MODEL=can RUN_NAME=can_c128 EXTRA="--base-channels 128" sbatch job_train.sh  # wider variant

#SBATCH --job-name=hipmri-seg
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=08:00:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

set -euo pipefail

MODEL=${MODEL:-can}
EPOCHS=${EPOCHS:-30}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
SEED=${SEED:-42}
RUN_NAME=${RUN_NAME:-$MODEL}
EXTRA=${EXTRA:-}
DATA_ROOT=/home/groups/comp3710/HipMRI_Study_open/keras_slices_data

source "$HOME/miniconda3/bin/activate"
conda activate torch

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "run=$RUN_NAME model=$MODEL epochs=$EPOCHS seed=$SEED ${LIMIT_ARG} ${EXTRA}"
nvidia-smi

python -u train.py \
    --model "$MODEL" \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --data-root "$DATA_ROOT" \
    --num-workers 2 \
    $LIMIT_ARG $EXTRA

echo "Finished $(date)"
