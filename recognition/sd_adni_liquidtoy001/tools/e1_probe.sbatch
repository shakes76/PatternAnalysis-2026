#!/bin/bash
#SBATCH --job-name=sd-e1-probe
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/e1_probe_%j.out
#SBATCH --error=logs/e1_probe_%j.err
# S1 of docs/e1-protocol.md: throughput, peak memory, weight save and reload.
# Probe weights go to job scratch, so they do not use home quota.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run="outputs/e1-probe-$SLURM_JOB_ID"
weights="${TMPDIR:?}/e1-probe-weights"
.venv-train/bin/python train.py --output "$run" --weights-dir "$weights" \
    --max-steps 200 --train-limit 3200 --val-images 256 --save-epochs
.venv-train/bin/python predict.py --weights "$weights/unet-ema-epoch-01.safetensors" \
    --output "$run/predict" --count 2
ls -la "$weights"
