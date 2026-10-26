#!/bin/bash
#SBATCH --job-name=hipmri-paired-training
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"

# One GPU, sequential models, identical patient manifest and training voxel weights.
# No test split is evaluated in this job.
target_epochs="${1:-30}"
if ! [[ "$target_epochs" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Pass a positive total epoch count.' >&2
  exit 2
fi
for model in unet2d improved3d; do
  batch_size=1
  if [[ "$model" == unet2d ]]; then batch_size=16; fi
  output_dir="artifacts/full_seed42/$model"
  resume_args=()
  if [[ -f "$output_dir/latest.pt" ]]; then
    completed_epochs=$(.venv/bin/python - "$output_dir/latest.pt" <<'PY'
import sys
import torch
print(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['epoch'])
PY
)
    if (( completed_epochs >= target_epochs )); then
      echo "$model already completed $completed_epochs epochs; target is $target_epochs."
      continue
    fi
    resume_args=(--resume "$output_dir/latest.pt")
  fi
  .venv/bin/python -u train.py --manifest artifacts/cache64/manifest.json \
    --model "$model" --output-dir "$output_dir" --epochs "$target_epochs" \
    --batch-size "$batch_size" --base-channels 16 --lr 0.001 \
    --workers 0 --seed 42 --device cuda --amp --eval-batch-size 32 \
    "${resume_args[@]}"
done
