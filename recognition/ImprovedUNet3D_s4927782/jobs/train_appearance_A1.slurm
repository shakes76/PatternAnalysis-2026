#!/bin/bash
#SBATCH --job-name=hipmri-appearance-A1
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/artifacts/appearance_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python -m unittest discover -s tests -p test_appearance.py -v
.venv/bin/python -m unittest discover -s tests -p test_evaluate_appearance.py -v
for variant in gamma_only gamma_polarity; do
  .venv/bin/python -u experiments/train_appearance.py \
    --protocol appearance_protocol.json --variant "$variant" --device cuda \
    --smoke-output "artifacts/appearance_smoke_${SLURM_JOB_ID}/$variant"
done
for variant in gamma_only gamma_polarity; do
  .venv/bin/python -u experiments/train_appearance.py \
    --protocol appearance_protocol.json --variant "$variant" --device cuda
done
.venv/bin/python -u experiments/evaluate_appearance.py \
  --protocol appearance_protocol.json --manifest artifacts/cache64/manifest.json \
  --output-dir "artifacts/appearance_evaluation_A1_${SLURM_JOB_ID}" --device cuda
