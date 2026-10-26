#!/bin/bash
#SBATCH --job-name=hipmri-validation
#SBATCH --account=comp3710
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

# Submit with afterok on the training job. Run sequentially on one GPU and
# inspect validation before fixing any final-test decision.
for model in unet2d improved3d; do
  .venv/bin/python tools/record_provenance.py --expected training_sources.json \
    --manifest artifacts/cache64/manifest.json \
    --output "artifacts/full_seed42/$model/source_provenance.json" \
    --checkpoint "artifacts/full_seed42/$model/best.pt" --replace
  .venv/bin/python -u predict.py --manifest artifacts/cache64/manifest.json \
    --checkpoint "artifacts/full_seed42/$model/best.pt" \
    --split validation --output-dir "artifacts/validation_seed42/$model" \
    --batch-size 32 --device cuda --target-class 5
done
