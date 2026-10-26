#!/bin/bash
#SBATCH --job-name=hipmri-pipeline
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
.venv/bin/python -m unittest discover -s tests -v

cache_dir="artifacts/cache_smoke_${SLURM_JOB_ID}"
run_dir="artifacts/pipeline_smoke_${SLURM_JOB_ID}"
.venv/bin/python tools/prepare_cache.py --audit artifacts/dataset_audit.json \
  --output-dir "$cache_dir" --shape 32 64 64 --label-ids 0 1 2 3 4 5 --limit 2
.venv/bin/python train.py --manifest "$cache_dir/manifest.json" \
  --model unet2d --output-dir "$run_dir/unet2d" \
  --epochs 1 --batch-size 4 --base-channels 4 --workers 0 \
  --max-batches 1 --device cpu
.venv/bin/python train.py --manifest "$cache_dir/manifest.json" \
  --model unet2d --output-dir "$run_dir/unet2d" \
  --epochs 2 --batch-size 4 --base-channels 4 --workers 0 \
  --max-batches 1 --device cpu --resume "$run_dir/unet2d/latest.pt"
.venv/bin/python train.py --manifest "$cache_dir/manifest.json" \
  --model improved3d --output-dir "$run_dir/improved3d" \
  --epochs 1 --batch-size 1 --base-channels 4 --workers 0 \
  --max-batches 1 --device cpu
