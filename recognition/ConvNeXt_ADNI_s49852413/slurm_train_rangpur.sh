#!/bin/bash
#SBATCH --job-name=adni_convnext_s49852413
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

# ==============================================================================
# UQ Rangpur HPC Slurm Batch Script — ADNI ConvNeXt Classification & Triage
# Author: Lingxiao Li (s49852413)
# Usage:  sbatch slurm_train_rangpur.sh
# ==============================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs images checkpoints

# Activate virtual environment or Conda environment on Rangpur
if [ -f "$HOME/comp3710-keras-demo/.venv/bin/activate" ]; then
    source "$HOME/comp3710-keras-demo/.venv/bin/activate"
elif [ -f ".venv/bin/activate" ]; then
    source ".venv/bin/activate"
fi

export ADNI_ROOT="${ADNI_ROOT:-/home/groups/comp3710/ADNI/AD_NC}"

echo "Running unit tests and AST compliance checks..."
python -m pytest test_pipeline.py -q

echo "Training Baseline SimpleCNN and Proposed ConvNeXt on ADNI ($ADNI_ROOT)..."
python -u train.py \
    --data_root "$ADNI_ROOT" \
    --model both \
    --image_size 128 \
    --batch_size 32 \
    --epochs 20 \
    --lr 8e-4 \
    --device cuda

echo "Running Test Inference & Failure Case Autopsy..."
python -u predict.py \
    --data_root "$ADNI_ROOT" \
    --triage_threshold 0.80 \
    --device cuda
