#!/bin/bash
#SBATCH --job-name=hipmri-training-pilot
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
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available()
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA:', torch.version.cuda, 'compiled architectures:', torch.cuda.get_arch_list())
PY
# Two complete training/validation epochs per configuration estimate cost and
# validate learning. These checkpoints can be resumed with identical settings.
for model in unet2d improved3d; do
  batch_size=1
  if [[ "$model" == unet2d ]]; then batch_size=16; fi
  .venv/bin/python -u train.py --manifest artifacts/cache64/manifest.json \
    --model "$model" --output-dir "artifacts/full_seed42/$model" --epochs 2 \
    --batch-size "$batch_size" --base-channels 16 --lr 0.001 \
    --workers 0 --seed 42 --device cuda --amp --eval-batch-size 32
done
