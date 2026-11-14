#!/bin/bash
#SBATCH --job-name=sd-audits
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/audits_%j.out
#SBATCH --error=logs/audits_%j.err
# CPU audits for any cohorts: pixel nearest neighbours (memorisation), appearance
# statistics and private closest-pair panels, as used for E1 to E3.
# Submit from the project folder. Usage: sbatch tools/cohort_audits.sbatch NAME=PATH ...
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
[ "$#" -gt 0 ] || { echo "at least one NAME=PATH cohort is required" >&2; exit 2; }
paths=()
extras=()
for item in "$@"; do
    paths+=("${item#*=}")
    extras+=(--extra "$item")
done
neighbours="outputs/neighbours-$SLURM_JOB_ID"
.venv-train/bin/python tools/audit_cohort.py "${paths[@]}" --output "$neighbours"
.venv-train/bin/python tools/audit_appearance.py --real \
    --baseline ../runs/baseline128-20261005/outputs/baseline-vae-632570 \
    "${extras[@]}" --output "outputs/appearance-$SLURM_JOB_ID"
.venv-train/bin/python tools/closest_panels.py "$neighbours/private-neighbors.json" \
    --output "outputs/closest-$SLURM_JOB_ID" --count 5
