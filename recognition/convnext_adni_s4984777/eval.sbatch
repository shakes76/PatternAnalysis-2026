#!/bin/bash
#SBATCH --job-name=adni_evaluate
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=evaluate_%j.out
#SBATCH --error=evaluate_%j.err

set -e

cd /home/Student/s4984777/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777

echo "======================================="
echo "COMP3710 - Final Model Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================="

nvidia-smi

echo "Evaluating BaselineCNN..."

python -u evaluate.py \
    --model baseline \
    --checkpoint checkpoints/baseline/best.pth \
    --output-dir results/baseline \
    --batch-size 8 \
    --require-cuda

echo "Evaluating ConvNeXt-Tiny..."

python -u evaluate.py \
    --model convnext \
    --checkpoint checkpoints/convnext/best.pth \
    --output-dir results/convnext \
    --batch-size 8 \
    --require-cuda

echo "======================================="
echo "ALL EVALUATIONS COMPLETED"
echo "======================================="

