#!/bin/bash
#SBATCH --job-name=hipmri-recovery-M2
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -m unittest discover -s tests -p test_evaluate_recovery_m015.py -v
.venv/bin/python -m unittest discover -s tests -p test_summarize_recovery_m015.py -v
.venv/bin/python -u tools/evaluate_recovery_m015.py \
  --protocol recovery_m015_protocol.json \
  --manifest artifacts/cache64/manifest.json \
  --output-dir "artifacts/recovery_m015_M2_${SLURM_JOB_ID}"
