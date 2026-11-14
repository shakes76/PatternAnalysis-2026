#!/bin/bash
#SBATCH --job-name=sd-b3
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=logs/b3_train_%j.out
#SBATCH --error=logs/b3_train_%j.err
# S2 of docs/b3-ldm-scratch-protocol.md. The epoch count comes from the probe by the
# protocol's rule and is passed at submission, e.g. --epochs 121 --save-epochs 60 121.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python train.py --output "outputs/b3-$SLURM_JOB_ID" --time-budget-minutes 160 \
    --unet-init scratch --lr 1e-4 --warmup-steps 1000 --batch-size 64 \
    --exclude-defective-scans --expect-excluded 2 40 "$@"
