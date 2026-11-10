#!/bin/bash
#SBATCH --job-name=sd-pilot200
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/train_pilot_%j.out
#SBATCH --error=logs/train_pilot_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir="outputs/pilot-$SLURM_JOB_ID"
mkdir -p "$run_dir"
.venv-train/bin/python tools/train_pilot.py --steps 100 --output "$run_dir/first100"
.venv-train/bin/python tools/train_pilot.py --steps 200 --output "$run_dir/final200" \
    --resume "$run_dir/first100/checkpoint.pt"
