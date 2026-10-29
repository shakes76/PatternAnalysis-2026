#!/bin/bash
#SBATCH --job-name=convnext_train
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=convnext_%j.out
#SBATCH --error=convnext_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd ~/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777

echo "Starting ConvNeXt-Tiny training"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

nvidia-smi

python -u train.py \
    --model convnext \
    --epochs 5 \
    --batch-size 8 \
    --lr 0.0001 \
    --require-cuda

echo "Training completed successfully!"

