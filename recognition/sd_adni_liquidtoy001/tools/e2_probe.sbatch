#!/bin/bash
#SBATCH --job-name=sd-e2-probe
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/e2_probe_%j.out
#SBATCH --error=logs/e2_probe_%j.err
# S1 of docs/e2-protocol.md: the exclusion count on the full training split,
# throughput, and weight save and reload. Probe weights go to job scratch.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run="outputs/e2-probe-$SLURM_JOB_ID"
weights="${TMPDIR:?}/e2-probe-weights"
.venv-train/bin/python train.py --output "$run" --weights-dir "$weights" \
    --max-steps 200 --val-images 256 --save-epochs \
    --exclude-defective-scans --expect-excluded 2 40
.venv-train/bin/python predict.py --weights "$weights/unet-ema-epoch-01.safetensors" \
    --output "$run/predict" --count 2
ls -la "$weights"
