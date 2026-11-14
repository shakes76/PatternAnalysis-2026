#!/bin/bash
#SBATCH --job-name=sd-b2
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/b2_train_%j.out
#SBATCH --error=logs/b2_train_%j.err
# S2 of docs/b2-gan-protocol.md: 30,000 steps with the collapse rule from step 5,000.
# No --mem directive: on Rangpur one makes the job pend.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python tools/baseline_gan.py train --output "outputs/b2-$SLURM_JOB_ID" \
    --steps 30000 --max-minutes 105 --expect-excluded 2 40
