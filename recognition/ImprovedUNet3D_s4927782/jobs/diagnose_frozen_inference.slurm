#!/bin/bash
#SBATCH --job-name=hipmri-frozen-diagnosis
#SBATCH --account=comp3710
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
# Frozen weights and explicit cases: diagnostic forward parity only.
# No optimizer, selection, aggregate ranking, postprocessing or overwritten arrays.
.venv/bin/python tools/diagnose_frozen_inference.py \
  --manifest artifacts/cache64/manifest.json \
  --baseline-checkpoint artifacts/full_seed42/unet2d/best.pt \
  --model-checkpoint artifacts/full_seed42/improved3d/best.pt \
  --baseline-eval artifacts/test_final/unet2d/evaluation.json \
  --baseline-eval artifacts/validation_seed42/unet2d/evaluation.json \
  --model-eval artifacts/test_final/improved3d/evaluation.json \
  --model-eval artifacts/validation_seed42/improved3d/evaluation.json \
  --case B006/Week0 --case K019/Week1 \
  --case B040/Week0 --case K018/Week0 \
  --case M015/Week0 --case M015/Week1 \
  --case M015/Week2 --case M015/Week3 \
  --case M015/Week4 --case M015/Week5 \
  --case M015/Week6 --case M015/Week7 \
  --diagnostic-test-ack --device cuda \
  --output-dir "artifacts/diagnostic_forward_${SLURM_JOB_ID}"
