#!/bin/bash
#SBATCH --job-name=hipmri-seed43-validation
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
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

# A second paired seed on TRAIN/VAL only. Never changes or reevaluates v1 TEST.
# Verify the follow-up plan and byte identities before loading any model.
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib, json
plan = json.loads(Path('followup_protocol.json').read_text())
if plan['experiment_id'] != 'paired_seed43_validation' or plan['training_seed'] != 43:
    raise SystemExit('Unexpected follow-up protocol identity.')
for relative, expected in plan['required_file_sha256'].items():
    if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Frozen follow-up source differs: {relative}')
if not plan['diagnostic_gate']['approved_to_proceed']:
    raise SystemExit('Diagnosis must be reviewed before follow-up training.')
PY

for model in unet2d improved3d; do
  batch_size=1
  if [[ "$model" == unet2d ]]; then batch_size=16; fi
  output_dir="artifacts/followup_seed43/$model"
  resume_args=()
  needs_training=1
  if [[ -f "$output_dir/latest.pt" ]]; then
    completed_epochs=$(.venv/bin/python - "$output_dir/latest.pt" "$model" <<'PY'
from pathlib import Path
import json, sys, torch
plan = json.loads(Path('followup_protocol.json').read_text())
checkpoint = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
expected = dict(plan['training_conditions'][sys.argv[2]], seed=43)
if checkpoint['config'] != expected or not 1 <= checkpoint['epoch'] <= 30:
    raise SystemExit('Existing follow-up checkpoint has unexpected config/epoch.')
if checkpoint['epoch'] == 30:
    run = json.loads((Path(sys.argv[1]).parent / 'run.json').read_text())
    if run['status'] != 'completed' or run['completed_epochs'] != 30 or run['config'] != expected:
        raise SystemExit('Existing completed run does not agree with checkpoint.')
print(checkpoint['epoch'])
PY
)
    if (( completed_epochs == 30 )); then
      echo "$model already completed $completed_epochs epochs."
      needs_training=0
    else
      resume_args=(--resume "$output_dir/latest.pt")
    fi
  fi
  if (( needs_training )); then
    .venv/bin/python -u train.py --manifest artifacts/cache64/manifest.json \
      --model "$model" --output-dir "$output_dir" --epochs 30 \
      --batch-size "$batch_size" --base-channels 16 --lr 0.001 \
      --workers 0 --seed 43 --device cuda --amp --eval-batch-size 32 \
      "${resume_args[@]}"
  fi
  .venv/bin/python tools/record_provenance.py --expected training_sources.json \
    --manifest artifacts/cache64/manifest.json \
    --output "$output_dir/source_provenance.json" \
    --checkpoint "$output_dir/best.pt" --replace
  if [[ ! -f "artifacts/validation_seed43/$model/evaluation.json" ]]; then
    .venv/bin/python -u predict.py --manifest artifacts/cache64/manifest.json \
      --checkpoint "$output_dir/best.pt" --split validation \
      --output-dir "artifacts/validation_seed43/$model" \
      --batch-size 32 --device cuda --target-class 5
  fi
  .venv/bin/python - "$model" <<'PY'
from pathlib import Path
import hashlib, json, sys
plan = json.loads(Path('followup_protocol.json').read_text())
model = sys.argv[1]
run_dir = Path('artifacts/followup_seed43') / model
evaluation = json.loads((Path('artifacts/validation_seed43') / model / 'evaluation.json').read_text())
expected = dict(plan['training_conditions'][model], seed=43)
if (evaluation['split'] != 'validation' or evaluation['final_test_explicitly_requested']
        or evaluation['development_only'] or evaluation['model_config'] != expected
        or evaluation['manifest_sha256'] != plan['manifest_sha256']
        or evaluation['checkpoint_sha256'] != hashlib.sha256((run_dir / 'best.pt').read_bytes()).hexdigest()):
    raise SystemExit('Follow-up validation identity differs from the frozen plan/checkpoint.')
print(f'{model}: follow-up validation identity verified.')
PY
done
