#!/bin/bash
#SBATCH --job-name=flan_t5_lora
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --output=flan_t5_%j.out
#SBATCH --error=flan_t5_%j.err

set -e

source "$HOME/miniconda3/bin/activate"
conda activate flan-t5

cd "$SLURM_SUBMIT_DIR"

python -u train.py \
    --mode lora \
    --epochs 2 \
    --batch-size 4 \
    --grad-accum 4 \
    --output-dir outputs/lora

echo "LoRA training completed."

