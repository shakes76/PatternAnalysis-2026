#!/bin/bash
#SBATCH --job-name=hipmri-independent-recovery
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/artifacts/pycache"
export HIPMRI_V1_PROJECT_ROOT=/home/Student/s4927782/comp3710-demo3
DEMO3_PYTHON="$HIPMRI_V1_PROJECT_ROOT/.venv/bin/python"
"$DEMO3_PYTHON" -m unittest -v test_parallel_prepare_v2.py
"$DEMO3_PYTHON" -u parallel_prepare_v2.py \
  --project-root "$HIPMRI_V1_PROJECT_ROOT" --protocol protocol_v2.json \
  --original-manifest "$HIPMRI_V1_PROJECT_ROOT/artifacts/cache64/manifest.json" \
  --raw-root /home/groups/comp3710/HipMRI_Study_open \
  --output-dir artifacts/cache_v2_recovered --workers 4 \
  --amendment cache_recovery_amendment.json
