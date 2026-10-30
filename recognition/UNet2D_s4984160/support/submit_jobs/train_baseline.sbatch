#!/bin/bash
#SBATCH --job-name=unet2d_baseline
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=01:30:00
#SBATCH --output=unet2d_%j.out
#SBATCH --error=unet2d_%j.err
set -eo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/PatternAnalysis-2026/recognition/UNet2D_s4984160"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
python -m unittest discover -s support/check_code -v
echo "Two real-data batches: execution check only, not performance evidence"
python -u train.py \
    --data-dir /home/groups/comp3710/HipMRI_Study_open \
    --output-dir "results/smoke_${SLURM_JOB_ID}" \
    --device cuda --threads 1 --workers 0 \
    --epochs 1 --smoke-steps 2 --batch-size 16 --base-features 16
echo "Full training: 8700 training slices, 2340 validation slices; test not evaluated"
python -u train.py \
    --data-dir /home/groups/comp3710/HipMRI_Study_open \
    --output-dir "results/baseline2d_${SLURM_JOB_ID}" \
    --device cuda --threads 1 --workers 0 \
    --epochs 30 --batch-size 16 --base-features 16 \
    --lr 0.001 --dice-weight 0.5 --min-epochs 10 --patience 8
echo "Finish: $(date)"
