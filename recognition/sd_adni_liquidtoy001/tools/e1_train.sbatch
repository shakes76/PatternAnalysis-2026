#!/bin/bash
#SBATCH --job-name=sd-e1
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00
#SBATCH --output=logs/e1_train_%j.out
#SBATCH --error=logs/e1_train_%j.err
# S2 of docs/e1-protocol.md. No --mem directive: on Rangpur one makes the job pend.
# Extra arguments are passed to train.py, e.g. for the LoRA ablation.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python -c 'import shutil; assert shutil.disk_usage(".").free > 5 * 1024**3, "Need 5 GiB free"'
.venv-train/bin/python train.py --output "outputs/e1-$SLURM_JOB_ID" --time-budget-minutes 200 "$@"
