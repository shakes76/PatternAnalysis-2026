#!/bin/bash
#SBATCH --job-name=convnext_frozen
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=exp3_%j.out
#SBATCH --error=exp3_%j.err

set -e

# Navigate to project directory
cd /home/Student/s4984777/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777

echo "========================================"
echo "Experiment 3: Frozen ConvNeXt-Tiny"
echo "Learning rate: 0.001"
echo "Max epochs: 10"
echo "Batch size: 8"
echo "Early stopping patience: 3"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================"

# Check GPU availability
nvidia-smi

# Start training
python -u train.py \
    --model convnext \
    --epochs 10 \
    --batch-size 8 \
    --lr 0.001 \
    --freeze-backbone \
    --early-stopping-patience 3 \
    --checkpoint-dir checkpoints/exp3_frozen \
    --require-cuda

echo "========================================"
echo "EXPERIMENT 3 COMPLETED SUCCESSFULLY"
echo "========================================"

