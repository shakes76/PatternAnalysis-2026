#!/bin/bash
#SBATCH --job-name=convnext_partial
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=exp4_%j.out
#SBATCH --error=exp4_%j.err

set -e

cd /home/Student/s4984777/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777

echo "Experiment 4: Partial ConvNeXt Fine-Tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

nvidia-smi

python -u train.py \
    --model convnext \
    --epochs 10 \
    --batch-size 8 \
    --lr 0.00001 \
    --partial-finetune \
    --early-stopping-patience 3 \
    --checkpoint-dir checkpoints/exp4_partial \
    --require-cuda

echo "EXPERIMENT 4 COMPLETED SUCCESSFULLY"

