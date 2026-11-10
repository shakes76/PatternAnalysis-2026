#!/bin/bash
#SBATCH --job-name=sd-e3-audits
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/e3_audits_%j.out
#SBATCH --error=logs/e3_audits_%j.err
# S3 of docs/e3-protocol.md, CPU part: pixel nearest neighbours, appearance
# statistics and private closest-pair panels, as for E1 and E2.
# Submit from the project folder. Usage: sbatch tools/e3_audits.sbatch EPOCH10 EPOCH05
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
e10="${1:?epoch-10 cohort folder}"
e05="${2:?epoch-5 cohort folder}"
neighbours="outputs/e3-neighbours-$SLURM_JOB_ID"
.venv-train/bin/python tools/audit_cohort.py "$e10" "$e05" --output "$neighbours"
.venv-train/bin/python tools/audit_appearance.py --real \
    --baseline ../runs/baseline128-20261005/outputs/baseline-vae-632570 \
    --extra "e3e10=$e10" --extra "e3e05=$e05" \
    --output "outputs/e3-appearance-$SLURM_JOB_ID"
.venv-train/bin/python tools/closest_panels.py "$neighbours/private-neighbors.json" \
    --output "outputs/e3-closest-$SLURM_JOB_ID" --count 5
