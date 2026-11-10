#!/bin/bash
#SBATCH --job-name=sd-e1-audits
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/e1_audits_%j.out
#SBATCH --error=logs/e1_audits_%j.err
# S3 of docs/e1-protocol.md, CPU part: pixel nearest neighbours against training
# and validation (memorisation audit) and the appearance statistics.
# Submit from the project folder. Usage: sbatch tools/e1_audits.sbatch EPOCH10 EPOCH05
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
e10="${1:?epoch-10 cohort folder}"
e05="${2:?epoch-5 cohort folder}"
.venv-train/bin/python tools/audit_cohort.py "$e10" "$e05" \
    --output "outputs/e1-neighbours-$SLURM_JOB_ID"
.venv-train/bin/python tools/audit_appearance.py --real \
    --baseline ../runs/baseline128-20261005/outputs/baseline-vae-632570 \
    --extra "e1e10=$e10" --extra "e1e05=$e05" \
    --output "outputs/e1-appearance-$SLURM_JOB_ID"
