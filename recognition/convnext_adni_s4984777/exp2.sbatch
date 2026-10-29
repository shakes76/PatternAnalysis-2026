#!/bin/bash
#SBATCH --job-name=convnext_lr1e5
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=exp2_%j.out
#SBATCH --error=exp2_%j.err

set -e

cd /home/Student/s4984777/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777

echo "================================"
echo "Experiment 2: ConvNeXt-Tiny"
echo "Learning rate: 1e-5"
echo "Epochs: 10"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "================================"

nvidia-smi

python -u train.py \
    --model convnext \
    --epochs 10 \
    --batch-size 8 \
    --lr 0.00001 \
    --checkpoint-dir checkpoints/exp2_lr1e-5 \
    --require-cuda

echo "EXPERIMENT 2 COMPLETED SUCCESSFULLY"

