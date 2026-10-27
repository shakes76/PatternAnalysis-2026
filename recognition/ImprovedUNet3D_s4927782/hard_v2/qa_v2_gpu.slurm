#!/bin/bash
#SBATCH --job-name=hipmri-independent-fit
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/artifacts/pycache"
export HIPMRI_V1_PROJECT_ROOT=/home/Student/s4927782/comp3710-demo3
"$HIPMRI_V1_PROJECT_ROOT/.venv/bin/python" -u qa_v2.py \
  --project-root "$HIPMRI_V1_PROJECT_ROOT" --protocol protocol_v2.json \
  --manifest artifacts/cache_v2/manifest.json \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --original-through artifacts/through_704494/pipeline_receipt.json \
  --output-dir artifacts/qa_v2
