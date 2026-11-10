#!/bin/bash
#SBATCH --job-name=sd-e3
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=logs/e3_train_%j.out
#SBATCH --error=logs/e3_train_%j.err
# S2 of docs/e3-protocol.md: E2's command with the rank-4 adapter at lr 1e-4.
# No --mem directive: on Rangpur one makes the job pend.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python -c 'import shutil; assert shutil.disk_usage(".").free > 1024**3, "Need 1 GiB free"'
.venv-train/bin/python train.py --output "outputs/e3-$SLURM_JOB_ID" --time-budget-minutes 75 \
    --trainable lora --lora-rank 4 --lr 1e-4 \
    --exclude-defective-scans --expect-excluded 2 40
