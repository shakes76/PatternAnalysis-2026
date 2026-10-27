#!/usr/bin/env bash
# Same fixed TRAIN/VAL-only C/D experiment on one allocated GPU; no provisioning.
# PYTHON_BIN=/workspace/venv/bin/python bash jobs/train_recovery_R1_cloud.sh \
#   --protocol recovery_protocol.json
set -euo pipefail
RECOVERY_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$RECOVERY_PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RECOVERY_PROTOCOL="recovery_protocol.json"
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--protocol" ]]; then
    echo 'Usage: train_recovery_R1_cloud.sh [--protocol recovery_protocol.json]' >&2
    exit 2
  fi
  RECOVERY_PROTOCOL="$2"
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit('Recovery requires exactly one visible allocated CUDA GPU.')
print('Recovery GPU:', torch.cuda.get_device_name(0), '; PyTorch:', torch.__version__, flush=True)
PY
RECOVERY_RUN_ID="${RECOVERY_RUN_ID:-cloud_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ ! "$RECOVERY_RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo 'RECOVERY_RUN_ID must contain only letters, digits, underscore or hyphen.' >&2
  exit 2
fi
export MPLCONFIGDIR="$RECOVERY_PROJECT_ROOT/artifacts/recovery_mpl_${RECOVERY_RUN_ID}"
mkdir -p "$MPLCONFIGDIR"
"$PYTHON_BIN" -m unittest discover -s tests -p test_recovery.py -v
"$PYTHON_BIN" -m unittest discover -s tests -p test_evaluate_recovery.py -v
for variant in clean_adapt polarity_rehearsal; do
  "$PYTHON_BIN" -u experiments/train_recovery.py \
    --protocol "$RECOVERY_PROTOCOL" --variant "$variant" --device cuda \
    --smoke-output "artifacts/recovery_smoke_${RECOVERY_RUN_ID}/$variant"
done
for variant in clean_adapt polarity_rehearsal; do
  "$PYTHON_BIN" -u experiments/train_recovery.py \
    --protocol "$RECOVERY_PROTOCOL" --variant "$variant" --device cuda
done

"$PYTHON_BIN" -u experiments/evaluate_recovery.py \
  --protocol "$RECOVERY_PROTOCOL" --manifest artifacts/cache64/manifest.json \
  --output-dir "artifacts/recovery_evaluation_R1_${RECOVERY_RUN_ID}" --device cuda
