#!/bin/bash
#SBATCH --job-name=final-real
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_real_%j.out
#SBATCH --error=logs/final_real_%j.err
# Stage 2 of docs/final-evaluation-protocol.md: real-slice features and distances.
# Usage: sbatch tools/final_real.sbatch rehearsal
#        sbatch tools/final_real.sbatch test FROZEN_COMMIT   (reads the test split)
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
mode="${1:?rehearsal or test}"
frozen=()
[ "$#" -ge 2 ] && frozen=(--frozen-commit "$2")
.venv-train/bin/python tools/final_real_features.py --mode "$mode" "${frozen[@]}" \
    --output "outputs/final-real-$mode-$SLURM_JOB_ID"
