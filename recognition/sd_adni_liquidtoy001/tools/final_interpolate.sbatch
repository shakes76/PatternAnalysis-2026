#!/bin/bash
#SBATCH --job-name=final-interp
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_interpolate_%j.out
#SBATCH --error=logs/final_interpolate_%j.err
# I1 and I2 of docs/final-evaluation-protocol.md for one fine-tuned model.
# Usage: sbatch tools/final_interpolate.sbatch MODEL REAL_DIR rehearsal
#        sbatch tools/final_interpolate.sbatch MODEL REAL_DIR test FROZEN_COMMIT
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/final_models.sh
name="${1:?model required}"
real="${2:?real features folder required}"
mode="${3:?rehearsal or test}"
frozen=()
[ "$#" -ge 4 ] && frozen=(--frozen-commit "$4")
.venv-train/bin/python tools/interpolate_final.py --model "$name" --weights "${WEIGHTS[$name]}" \
    --real "$real" --exclude-scans "$EXCLUDED_SCANS" --mode "$mode" "${frozen[@]}" \
    --output "outputs/final-interp-$name-$mode-$SLURM_JOB_ID"
