#!/bin/bash
#SBATCH --job-name=sd-b2b-probe
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/b2b_probe_%j.out
#SBATCH --error=logs/b2b_probe_%j.err
# S1 of docs/b2b-gan-per-class-protocol.md: 500 steps of the AD-only GAN, collapse
# checks from step 0, and a cohort from the per-class generation path.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run="outputs/b2b-probe-$SLURM_JOB_ID"
.venv-train/bin/python tools/baseline_gan.py train --output "$run" --train-class AD \
    --steps 500 --sample-every 250 --collapse-after 0 --expect-excluded 2 40
.venv-train/bin/python tools/baseline_gan.py generate --output "$run/cohort" --count 2 \
    --class-checkpoint "AD=$run/final.pt" --class-checkpoint "NC=$run/final.pt"
