#!/bin/bash
#SBATCH --job-name=hipmri-final-test
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

# Check the validation-frozen identities before opening any final-test cases.
.venv/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path

protocol = json.loads(Path('test_protocol.json').read_text())
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
if protocol['test_evaluated_at_freeze'] is not False:
    raise ValueError('Expected a protocol frozen before test evaluation.')
if protocol['manifest_sha256'] != digest('artifacts/cache64/manifest.json'):
    raise ValueError('Frozen manifest identity differs.')
for model in ('unet2d', 'improved3d'):
    selected = protocol['models'][model]
    if (selected['checkpoint'] != f'artifacts/full_seed42/{model}/best.pt'
            or selected['checkpoint_sha256'] != digest(selected['checkpoint'])):
        raise ValueError('Frozen checkpoint identity differs.')
    target = Path('artifacts/test_final') / model / 'evaluation.json'
    if target.exists():
        raise ValueError('A final evaluation already exists; do not repeat it.')
print('Frozen manifest and both checkpoint identities verified.')
PY

for model in unet2d improved3d; do
  .venv/bin/python -u predict.py --manifest artifacts/cache64/manifest.json \
    --checkpoint "artifacts/full_seed42/$model/best.pt" \
    --split test --final-test --output-dir "artifacts/test_final/$model" \
    --batch-size 32 --device cuda --target-class 5
done
