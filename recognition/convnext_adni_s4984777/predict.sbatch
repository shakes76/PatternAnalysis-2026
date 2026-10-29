#!/bin/bash
#SBATCH --job-name=adni_predict
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=predict_%j.out
#SBATCH --error=predict_%j.err

set -e

cd /home/Student/s4984777/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777

python -u predict.py \
    --model convnext \
    --checkpoint checkpoints/convnext/best.pth \
    --test-index 0 \
    --output results/prediction_convnext.png \
    --require-cuda

echo "PREDICTION COMPLETED"

