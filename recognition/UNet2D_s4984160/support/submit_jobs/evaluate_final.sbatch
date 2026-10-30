#!/bin/bash
#SBATCH --job-name=unet2d_final_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=01:30:00
#SBATCH --output=unet2d_eval_%j.out
#SBATCH --error=unet2d_eval_%j.err
set -eo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/PatternAnalysis-2026/recognition/UNet2D_s4984160"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
python -u evaluate.py \
    --checkpoint results/frozen_epoch5_716367/best_model.pt \
    --expected-sha256 3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b \
    --reference-checkpoint ../UNet3D_s4984160/results/refine_v2/best_model.pt \
    --reference-sha256 77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156 \
    --reference-code-dir ../UNet3D_s4984160 \
    --reference-results ../UNet3D_s4984160/results/final_test_refine_v2 \
    --split-manifest support/patient_split.json \
    --threads 1 --batch-size 16 \
    --output-dir "results/final_test_${SLURM_JOB_ID}"
python -u predict.py render \
    --input-dir "results/final_test_${SLURM_JOB_ID}/prediction_inputs" \
    --output-dir "results/final_test_${SLURM_JOB_ID}/figures"
echo "Finish: $(date)"
