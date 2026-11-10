#!/bin/bash
#SBATCH --job-name=sd-e3-probe
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/e3_probe_%j.out
#SBATCH --error=logs/e3_probe_%j.err
# S1 of docs/e3-protocol.md: rank-4 adapter with E2's recipe for 200 steps,
# exclusion count, adapter save and reload. Probe weights go to job scratch.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run="outputs/e3-probe-$SLURM_JOB_ID"
weights="${TMPDIR:?}/e3-probe-weights"
.venv-train/bin/python train.py --output "$run" --weights-dir "$weights" \
    --trainable lora --lora-rank 4 --lr 1e-4 \
    --max-steps 200 --val-images 256 --save-epochs \
    --exclude-defective-scans --expect-excluded 2 40
.venv-train/bin/python predict.py --weights "$weights/lora-ema-epoch-01.safetensors" \
    --output "$run/predict" --count 2
ls -la "$weights"
