#!/bin/bash
# Slurm job for Rangpur (COMP3710 partition).
#   sbatch job.sh 2          # train 2D U-Net
#   sbatch job.sh 3          # train 3D U-Net
#   sbatch job.sh predict    # comparison table, base-apex plot, failure figures
# Extra arguments are passed straight to the Python script.
#SBATCH --job-name=hipmri_unet
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=04:00:00
#SBATCH --output=hipmri_%j.out
#SBATCH --error=hipmri_%j.err

MODE=${1:-2}
shift
ARGS=("$@")   # save the extra arguments for python
set --        # clear them so conda's activate doesn't receive them

# Activate the conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

echo "host $(hostname) | mode $MODE | args ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv

if [ "$MODE" = "predict" ]; then
    python -u predict.py "${ARGS[@]}"
elif [ "$MODE" = "2" ]; then
    python -u train.py --dims 2 --epochs 30 --batch 16 "${ARGS[@]}"
else
    python -u train.py --dims 3 --epochs 60 --batch 2 "${ARGS[@]}"
fi
