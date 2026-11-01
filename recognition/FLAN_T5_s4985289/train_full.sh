#!/bin/bash
#SBATCH --job-name=flan_t5_full
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=full_train_%j.out
#SBATCH --error=full_train_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate flan-t5

cd "$SLURM_SUBMIT_DIR"

echo "Starting full FLAN-T5 fine-tuning on $(hostname)"

python -u train.py \
    --mode full \
    --epochs 2 \
    --batch-size 4 \
    --grad-accum 4 \
    --seed 42 \
    --output-dir outputs/full

echo "Full fine-tuning completed."
