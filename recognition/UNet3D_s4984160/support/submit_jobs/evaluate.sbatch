#!/bin/bash
#SBATCH --job-name=unet3d_final_eval
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=01:30:00
#SBATCH --output=final_eval_%j.out
#SBATCH --error=final_eval_%j.err
set -eo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/PatternAnalysis-2026/recognition/UNet3D_s4984160"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
python -u evaluate.py \
    --checkpoint results/refine_v2/best_model.pt \
    --split test \
    --split-manifest results/refine_v2/split.json \
    --threads 1 \
    --output-dir results/final_test_refine_v2
python -u predict.py \
    --input-dir results/final_test_refine_v2/prediction_inputs \
    --output-dir results/final_test_refine_v2/figures
echo "Finish: $(date)"
