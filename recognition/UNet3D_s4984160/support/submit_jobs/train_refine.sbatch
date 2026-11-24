#!/bin/bash
#SBATCH --job-name=unet3d_refine
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=01:30:00
#SBATCH --output=refine_%j.out
#SBATCH --error=refine_%j.err
set -eo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/PatternAnalysis-2026/recognition/UNet3D_s4984160"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
python -u train.py \
    --init-checkpoint results/scratch_fast_v1/last_model.pt \
    --epochs 100 \
    --lr 0.0005 \
    --threads 1 \
    --workers 0 \
    --output-dir results/refine_v2
echo "Finish: $(date)"
