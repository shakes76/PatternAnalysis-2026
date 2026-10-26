#!/bin/bash
#SBATCH --job-name=hipmri-source-D1
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=json.loads(Path('source_diagnostic_protocol.json').read_text())
for relative,wanted in p['required_file_sha256'].items():
    if hashlib.sha256(Path(relative).read_bytes()).hexdigest()!=wanted:
        raise SystemExit('Diagnostic input changed: '+relative)
PY
.venv/bin/python -m unittest discover -s tests -p test_source_inputs.py -v
.venv/bin/python -m unittest discover -s tests -p test_source_archive_identity.py -v
.venv/bin/python tools/verify_source_archive_identity.py \
  --manifest artifacts/cache64/manifest.json \
  --catalog artifacts/official_zip_catalog_D1.json \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --output "artifacts/source_container_identity_D1_${SLURM_JOB_ID}.json"
.venv/bin/python tools/diagnose_source_inputs.py \
  --manifest artifacts/cache64/manifest.json \
  --cache-identity artifacts/cache64/cache_identity.json \
  --training-sources training_sources.json \
  --data-root /home/groups/comp3710/HipMRI_Study_open \
  --output-dir "artifacts/source_inputs_D1_${SLURM_JOB_ID}" \
  --ack-retrospective-m015-test \
  --case M015/Week0 --case M015/Week1 --case M015/Week2 --case M015/Week3 \
  --case M015/Week4 --case M015/Week5 --case M015/Week6 --case M015/Week7 \
  --case B006/Week0 --case K019/Week1 --case B040/Week0 --case K018/Week0 \
  --case R016/Week7 --case T014/Week3
