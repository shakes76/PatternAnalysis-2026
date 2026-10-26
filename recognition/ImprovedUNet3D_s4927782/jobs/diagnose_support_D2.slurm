#!/bin/bash
#SBATCH --job-name=hipmri-support-D2
#SBATCH --account=comp3710
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
plan = json.loads(Path('support_intervention_protocol.json').read_text())
for relative, wanted in plan['required_file_sha256'].items():
    if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != wanted:
        raise SystemExit('D2 input changed: ' + relative)
PY
.venv/bin/python -m unittest discover -s tests -p test_support_intervention.py -v
.venv/bin/python tools/diagnose_support_intervention.py \
  --manifest artifacts/cache64/manifest.json \
  --cache-identity artifacts/cache64/cache_identity.json \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --d1-source-report artifacts/source_inputs_D1_702061/source_input_diagnostic.json \
  --output-dir "artifacts/support_intervention_D2_${SLURM_JOB_ID}" \
  --diagnostic-test-ack --device cuda
