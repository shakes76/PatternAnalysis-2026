#!/bin/bash
#SBATCH --job-name=flan_lora_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=eval_lora_%j.out
#SBATCH --error=eval_lora_%j.err

set -e

source "$HOME/miniconda3/bin/activate"
conda activate flan-t5

cd "$SLURM_SUBMIT_DIR"

echo "Starting LoRA validation evaluation..."

python -u predict.py \
    --mode lora \
    --checkpoint outputs/lora/best \
    --split validation \
    --batch-size 8 \
    --output-dir outputs/eval_lora_validation

echo "LoRA evaluation completed."
