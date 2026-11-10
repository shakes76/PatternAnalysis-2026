#!/bin/bash
#SBATCH --job-name=sd-full-gate
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/validate_full_%j.out
#SBATCH --error=logs/validate_full_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir="outputs/full-gate-$SLURM_JOB_ID"
mkdir -p "$run_dir"
.venv-train/bin/python tools/train_full.py --steps 8 --output "$run_dir/continuous"
.venv-train/bin/python tools/train_full.py --steps 4 --output "$run_dir/interrupted"
.venv-train/bin/python tools/train_full.py --steps 8 --output "$run_dir/resumed" \
    --resume "$run_dir/interrupted/checkpoint.pt"
.venv-train/bin/python tools/compare_smoke.py "$run_dir"
