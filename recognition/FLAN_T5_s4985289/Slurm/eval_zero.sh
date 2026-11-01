#!/bin/bash
#SBATCH --job-name=flan_zero_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=eval_zero_%j.out
#SBATCH --error=eval_zero_%j.err

set -e

# Activate Conda environment
source "$HOME/miniconda3/bin/activate"
conda activate flan-t5

# Enter project directory
cd "$SLURM_SUBMIT_DIR"

echo "Starting Zero-shot FLAN-T5 evaluation..."

python -u predict.py \
    --mode zero \
    --split validation \
    --batch-size 8 \
    --source-len 256 \
    --target-len 128 \
    --output-dir outputs/eval_zero_validation

echo "Zero-shot evaluation completed."
