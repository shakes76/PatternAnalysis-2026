#!/bin/bash
#SBATCH --job-name=unet_resource_check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=01:30:00
#SBATCH --output=resource_%j.out
#SBATCH --error=resource_%j.err
set -eo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/PatternAnalysis-2026/recognition/UNet2D_s4984160"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Job: $SLURM_JOB_ID"
echo "Start: $(date)"
python -u support/profile_inference.py \
    --evaluator-code-dir . \
    --reference-code-dir ../UNet3D_s4984160 \
    --checkpoint-2d results/frozen_epoch5_716367/best_model.pt \
    --checkpoint-3d ../UNet3D_s4984160/results/refine_v2/best_model.pt \
    --sha-2d 3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b \
    --sha-3d 77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156 \
    --split-manifest support/patient_split.json \
    --output-dir "results/resource_${SLURM_JOB_ID}" --batch-size 16 --warmups 3 --repeats 3
python -u ../UNet3D_s4984160/predict.py \
    --checkpoint ../UNet3D_s4984160/results/refine_v2/best_model.pt \
    --mri /home/groups/comp3710/HipMRI_Study_open/semantic_MRs/B037_Week0_LFOV.nii.gz \
    --mask /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/B037_Week0_SEMANTIC.nii.gz \
    --expected-sha256 77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156 \
    --device cuda --precision bf16 --threads 1 \
    --output-dir "results/predict_demo_${SLURM_JOB_ID}"
echo "Finish: $(date)"
