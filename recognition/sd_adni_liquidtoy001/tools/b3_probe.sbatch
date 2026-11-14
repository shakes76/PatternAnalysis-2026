#!/bin/bash
#SBATCH --job-name=sd-b3-probe
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/b3_probe_%j.out
#SBATCH --error=logs/b3_probe_%j.err
# S1 of docs/b3-ldm-scratch-protocol.md: 300 steps of the scratch UNet at batch 64,
# throughput for the step-count rule, weight save and reload.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run="outputs/b3-probe-$SLURM_JOB_ID"
weights="${TMPDIR:?}/b3-probe-weights"
.venv-train/bin/python train.py --output "$run" --weights-dir "$weights" \
    --unet-init scratch --lr 1e-4 --warmup-steps 1000 --batch-size 64 \
    --max-steps 300 --val-images 256 --save-epochs \
    --exclude-defective-scans --expect-excluded 2 40
.venv-train/bin/python predict.py --unet-init scratch \
    --weights "$weights/unet-ema-epoch-01.safetensors" --output "$run/predict" --count 2
