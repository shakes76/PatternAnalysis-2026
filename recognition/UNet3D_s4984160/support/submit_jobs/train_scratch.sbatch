#!/bin/bash
#SBATCH --job-name=unet3d_scratch_fast
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=01:30:00
#SBATCH --output=scratch_fast_%j.out
#SBATCH --error=scratch_fast_%j.err
set -eo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/PatternAnalysis-2026/recognition/UNet3D_s4984160"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
echo "Phase 1: two-case real-data execution check, NOT performance evidence"
python -u -m support.train_phase1.scratch --epochs 1 --threads 1 --workers 0 --max-train-batches 2 --max-val-cases 2 --output-dir results/scratch_fast_smoke
echo "Phase 2: full from-scratch training; 145 train and 39 validation scans"
python -u -m support.train_phase1.scratch --epochs 40 --lr 0.0005 --threads 1 --workers 0 --output-dir results/scratch_fast_v1
echo "Finish: $(date)"
