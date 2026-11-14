#!/bin/bash
#SBATCH --job-name=final-cohorts
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_cohorts_%j.out
#SBATCH --error=logs/final_cohorts_%j.err
# Stage 2 of docs/final-evaluation-protocol.md: features and nearest neighbours
# of generated cohorts against the real arrays of one final_real job.
# Usage: sbatch tools/final_cohorts.sbatch REAL_DIR NAME=PATH [NAME=PATH ...]
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
real="${1:?real features folder required}"
shift
[ "$#" -gt 0 ] || { echo "at least one NAME=PATH cohort is required" >&2; exit 2; }
cohorts=()
for item in "$@"; do
    cohorts+=(--cohort "$item")
done
.venv-train/bin/python tools/final_cohort_features.py --real "$real" "${cohorts[@]}" \
    --output "outputs/final-cohorts-$SLURM_JOB_ID"
