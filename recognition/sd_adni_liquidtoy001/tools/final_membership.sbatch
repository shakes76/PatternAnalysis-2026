#!/bin/bash
#SBATCH --job-name=final-mia
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_membership_%j.out
#SBATCH --error=logs/final_membership_%j.err
# M3 of docs/final-evaluation-protocol.md: loss-based membership scores of one
# diffusion model (sd is the control).
# Usage: sbatch tools/final_membership.sbatch MODEL rehearsal
#        sbatch tools/final_membership.sbatch MODEL test FROZEN_COMMIT
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
source tools/final_models.sh
name="${1:?model required}"
mode="${2:?rehearsal or test}"
frozen=()
[ "$#" -ge 3 ] && frozen=(--frozen-commit "$3")
diffusion_args "$name"
.venv-train/bin/python tools/membership.py --model "$name" "${model_args[@]}" \
    --exclude-scans "$EXCLUDED_SCANS" --mode "$mode" "${frozen[@]}" \
    --output "outputs/final-mia-$name-$mode-$SLURM_JOB_ID"
