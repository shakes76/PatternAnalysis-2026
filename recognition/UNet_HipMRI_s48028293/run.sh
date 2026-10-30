#!/bin/bash
# Usage:  sbatch run.sh unet2d             (2D U-Net)
#         sbatch run.sh unet3d             (3D U-Net)
#         sbatch run.sh improved_unet3d    (3D Improved U-Net)
#
# Default settings for each model are in MODEL_DEFAULTS in train.py.
#SBATCH --job-name=unet_hipmri
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err

MODEL=$1
if [ -z "$MODEL" ]; then
    echo "Error: give a model name, e.g. sbatch run.sh unet2d" >&2
    exit 1
fi
set --
set -e

source $HOME/miniconda3/bin/activate
conda activate torch

OUT=../runs/${MODEL}_$SLURM_JOB_ID

python train.py --model $MODEL --out-dir $OUT
python predict.py --model $MODEL --out-dir $OUT --checkpoint $OUT/best_model.pt