#!/bin/bash
#SBATCH --job-name=sd-b2b
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=logs/b2b_train_%j.out
#SBATCH --error=logs/b2b_train_%j.err
# S2 of docs/b2b-gan-per-class-protocol.md: one class's GAN, 20,000 steps, collapse
# rule from step 5,000. Usage: sbatch tools/b2b_train.sbatch AD   (or NC)
# No --mem directive: on Rangpur one makes the job pend.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
label="${1:?class required: AD or NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python tools/baseline_gan.py train --train-class "$label" \
    --output "outputs/b2b-$label-$SLURM_JOB_ID" \
    --steps 20000 --max-minutes 95 --expect-excluded 2 40
