#!/bin/bash
#SBATCH --job-name=final-profile
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_profile_%j.out
#SBATCH --error=logs/final_profile_%j.err
# Cost measures of docs/final-evaluation-protocol.md: sampling speed and memory.
# Usage: sbatch tools/final_profile.sbatch [--models e2 e3 b3 b1 b2]
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/final_models.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python tools/profile_sampling.py --e2 "${WEIGHTS[e2]}" --e3 "${WEIGHTS[e3]}" \
    --b3 "${WEIGHTS[b3]}" --b1 "$B1_CHECKPOINT" --b2 "$B2_CHECKPOINT" \
    --output "outputs/final-profile-$SLURM_JOB_ID" "$@"
