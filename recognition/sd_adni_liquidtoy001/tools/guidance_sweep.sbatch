#!/bin/bash
#SBATCH --job-name=guidance-sweep
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/guidance_sweep_%j.out
#SBATCH --error=logs/guidance_sweep_%j.err
# Post-hoc (tools/guidance_sweep.py): E2 at the guidance scales given, 128
# seeds per class; AUTOPSY=1 also regenerates the five autopsy seeds.
# Submitted from the final run directory (runs/final-20261019). Reads no
# test-split data. Usage: sbatch tools/guidance_sweep.sbatch 1 3
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/final_models.sh
[ "$#" -gt 0 ] || { echo "give at least one guidance scale" >&2; exit 2; }
autopsy=()
[ "${AUTOPSY:-0}" = 1 ] && autopsy=(--autopsy --reference-cohort outputs/final-e2)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python tools/guidance_sweep.py --weights "${WEIGHTS[e2]}" \
    --guidance "$@" "${autopsy[@]}" \
    --real-references outputs/final-real-rehearsal-672908 \
    --output "outputs/guidance-e2-$SLURM_JOB_ID"
