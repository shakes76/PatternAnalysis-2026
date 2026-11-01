#!/bin/bash
#SBATCH --job-name=flan_full_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=eval_full_%j.out
#SBATCH --error=eval_full_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate flan-t5

cd "$SLURM_SUBMIT_DIR"

echo "Starting Full FLAN-T5 validation evaluation..."

python -u predict.py \
    --mode full \
    --checkpoint outputs/full/best \
    --split validation \
    --batch-size 8 \
    --source-len 256 \
    --target-len 128 \
    --output-dir outputs/eval_full_validation

echo "Full evaluation completed."
