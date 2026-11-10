#!/bin/bash
#SBATCH --job-name=sd-e2-audits
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/e2_audits_%j.out
#SBATCH --error=logs/e2_audits_%j.err
# S3 of docs/e2-protocol.md, CPU part: pixel nearest neighbours (memorisation),
# appearance statistics and private closest-pair panels, as for E1.
# Submit from the project folder. Usage: sbatch tools/e2_audits.sbatch EPOCH10 EPOCH05
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
e10="${1:?epoch-10 cohort folder}"
e05="${2:?epoch-5 cohort folder}"
neighbours="outputs/e2-neighbours-$SLURM_JOB_ID"
.venv-train/bin/python tools/audit_cohort.py "$e10" "$e05" --output "$neighbours"
.venv-train/bin/python tools/audit_appearance.py --real \
    --baseline ../runs/baseline128-20261005/outputs/baseline-vae-632570 \
    --extra "e2e10=$e10" --extra "e2e05=$e05" \
    --output "outputs/e2-appearance-$SLURM_JOB_ID"
.venv-train/bin/python tools/closest_panels.py "$neighbours/private-neighbors.json" \
    --output "outputs/e2-closest-$SLURM_JOB_ID" --count 5
