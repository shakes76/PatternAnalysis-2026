#!/bin/bash
#SBATCH --job-name=HIPMRI_2D_ALIGNED
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=2D_ALIGNED_%j.out
#SBATCH --error=2D_ALIGNED_%j.err

set -e

source "$HOME/miniconda3/bin/activate"
conda activate torch

cd "$SLURM_SUBMIT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

nvidia-smi

# Default to 20 epochs, or use the supplied argument
EPOCHS="${1:-20}"

python -u - "$EPOCHS" <<'PY'
import sys
from train import train_2d_aligned

train_2d_aligned(
    epochs=int(sys.argv[1]),
    slice_batch_size=8
)
PY

echo "Finished: $(date)"

