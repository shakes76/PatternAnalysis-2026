#!/bin/bash
#SBATCH --job-name=sd-cohort-audit
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/cohort_%j.out
#SBATCH --error=logs/cohort_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
run_dir="outputs/cohort-$SLURM_JOB_ID"
mkdir -p "$run_dir"
checkpoint=../runs/full20k-20261004/outputs/full-629931/step-20000/checkpoint.pt
.venv-train/bin/python tools/generate_cohort.py --checkpoint "$checkpoint" --output "$run_dir/base" --base --align-scheduler-precision
.venv-train/bin/python tools/generate_cohort.py --checkpoint "$checkpoint" --output "$run_dir/adapted" --align-scheduler-precision
