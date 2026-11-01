#!/bin/bash
#SBATCH --job-name=clinical_audit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=clinical_eval_%j.out
#SBATCH --error=clinical_eval_%j.err

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate flan-t5

cd "$SLURM_SUBMIT_DIR"

echo "Evaluating Zero-shot..."
python -u predict.py \
    --mode zero \
    --split test \
    --limit 100 \
    --batch-size 8 \
    --source-len 256 \
    --target-len 128 \
    --output-dir outputs/clinical/zero

echo "Evaluating LoRA..."
python -u predict.py \
    --mode lora \
    --checkpoint outputs/lora/best \
    --split test \
    --limit 100 \
    --batch-size 8 \
    --source-len 256 \
    --target-len 128 \
    --output-dir outputs/clinical/lora

echo "Evaluating Full Fine-tuning..."
python -u predict.py \
    --mode full \
    --checkpoint outputs/full/best \
    --split test \
    --limit 100 \
    --batch-size 8 \
    --source-len 256 \
    --target-len 128 \
    --output-dir outputs/clinical/full

echo "Clinical evaluation completed."
