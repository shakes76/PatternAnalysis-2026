#!/bin/bash
#SBATCH --job-name=sd-vae-check
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --output=logs/vae_check_%j.out
#SBATCH --error=logs/vae_check_%j.err
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
.venv/bin/python tools/check_vae.py \
    --root /home/groups/comp3710/ADNI \
    --manifest data/manifest-seed3710.csv \
    --model models/sd15-vae \
    --output "outputs/vae-$SLURM_JOB_ID"
