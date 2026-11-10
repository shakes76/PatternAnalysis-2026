#!/bin/bash
#SBATCH --job-name=sd-b2-probe
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/b2_probe_%j.out
#SBATCH --error=logs/b2_probe_%j.err
# S1 of docs/b2-gan-protocol.md: 500 steps, throughput, the snapshot path
# (collapse checks from step 0 here), and a 4-image cohort from final.pt.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
run="outputs/b2-probe-$SLURM_JOB_ID"
.venv-train/bin/python tools/baseline_gan.py train --output "$run" \
    --steps 500 --sample-every 250 --collapse-after 0 --expect-excluded 2 40
.venv-train/bin/python tools/baseline_gan.py generate --checkpoint "$run/final.pt" \
    --output "$run/cohort" --count 2
