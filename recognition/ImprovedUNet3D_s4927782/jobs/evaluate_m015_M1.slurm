#!/bin/bash
#SBATCH --job-name=hipmri-m015-M1
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
.venv/bin/python -m unittest discover -s tests -p test_m015_followup.py -v
.venv/bin/python -u tools/evaluate_m015_followup.py \
  --protocol m015_followup_protocol.json \
  --manifest artifacts/cache64/manifest.json \
  --output-dir "artifacts/m015_followup_M1_${SLURM_JOB_ID}"
