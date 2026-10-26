#!/bin/bash
#SBATCH --job-name=hipmri-cache
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
# Optionally verify resume against an existing training-only smoke manifest.
smoke_cache="${1:-}"
if [[ -n "$smoke_cache" ]]; then
  smoke_run="artifacts/resume_check_${SLURM_JOB_ID}"
  .venv/bin/python train.py --manifest "$smoke_cache" --model unet2d \
    --output-dir "$smoke_run" --epochs 1 --batch-size 4 --base-channels 4 \
    --workers 0 --max-batches 1 --device cpu
  .venv/bin/python train.py --manifest "$smoke_cache" --model unet2d \
    --output-dir "$smoke_run" --epochs 2 --batch-size 4 --base-channels 4 \
    --workers 0 --max-batches 1 --device cpu --resume "$smoke_run/latest.pt"
  .venv/bin/python tools/inspect_cache.py --manifest "$smoke_cache" \
    --output artifacts/alignment_smoke.png --target-class 5
fi
.venv/bin/python tools/prepare_cache.py --audit artifacts/dataset_audit.json \
  --output-dir artifacts/cache64 --shape 64 128 128 --label-ids 0 1 2 3 4 5
.venv/bin/python tools/attach_label_schema.py \
  --manifest artifacts/cache64/manifest.json --schema label_schema.json
.venv/bin/python tools/inspect_cache.py --manifest artifacts/cache64/manifest.json \
  --output artifacts/alignment_full.png --target-class 5
