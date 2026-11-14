#!/bin/bash
#SBATCH --job-name=gan-noise
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/gan_noise_%j.out
#SBATCH --error=logs/gan_noise_%j.err
# Post-hoc (tools/gan_noise.py): how much the B2 and B2b generators' outputs
# depend on their noise, against real training slices. Submitted from the
# final run directory. Reads no test-split data.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
source tools/final_models.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
generators=(--generator "b2=$B2_CHECKPOINT")
[ -f "$B2B_AD" ] && generators+=(--generator "b2b-AD=$B2B_AD:AD")
[ -f "$B2B_NC" ] && generators+=(--generator "b2b-NC=$B2B_NC:NC")
.venv-train/bin/python tools/gan_noise.py "${generators[@]}" \
    --check-cohort outputs/final-b2-s37200-673110 \
    --real outputs/final-real-rehearsal-672908 --output "outputs/gan-noise-$SLURM_JOB_ID"
