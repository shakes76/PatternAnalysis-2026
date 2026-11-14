#!/bin/bash
#SBATCH --job-name=sd-full20k
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_full_%j.out
#SBATCH --error=logs/train_full_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-train/bin/python -c 'import shutil; assert shutil.disk_usage(".").free > 1024**3, "Need 1GiB free"'
run_dir="outputs/full-$SLURM_JOB_ID"
mkdir -p "$run_dir"
resume_args=()
for step in $(seq 2000 2000 20000); do
    stage="$run_dir/step-$step"
    .venv-train/bin/python tools/train_full.py --steps "$step" --output "$stage" "${resume_args[@]}"
    resume_args=(--resume "$stage/checkpoint.pt")
done
