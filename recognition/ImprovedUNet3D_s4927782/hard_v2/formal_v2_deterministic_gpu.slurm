#!/bin/bash
#SBATCH --job-name=hipmri-independent-pair-v2
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/artifacts/pycache"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/artifacts/matplotlib"
export HIPMRI_V1_PROJECT_ROOT=/home/Student/s4927782/comp3710-demo3
DEMO3_PYTHON="$HIPMRI_V1_PROJECT_ROOT/.venv/bin/python"
mkdir artifacts/formal_v2_execution.lock
"$DEMO3_PYTHON" - <<'PY'
from pathlib import Path
import v2_pipeline as v2
p=v2.load_protocol('protocol_v2.json')
v2.verify_review_sources(p,Path('.'))
v2.validate_qa_receipt('artifacts/qa_v2/qa_receipt.json','artifacts/cache_v2/manifest.json',p)
PY
for model in unet2d improved3d; do
  "$DEMO3_PYTHON" -u v2_pipeline.py --mode train --protocol protocol_v2.json \
    --project-root "$HIPMRI_V1_PROJECT_ROOT" --manifest artifacts/cache_v2/manifest.json \
    --model "$model" --output-dir "artifacts/formal_v2/$model" \
    --qa-receipt artifacts/qa_v2/qa_receipt.json --device cuda
  "$DEMO3_PYTHON" -u evaluate_v2_deterministic.py --mode evaluate --protocol protocol_v2.json \
    --execution-amendment engineering_execution_amendment.json \
    --project-root "$HIPMRI_V1_PROJECT_ROOT" --manifest artifacts/cache_v2/manifest.json \
    --model "$model" --checkpoint "artifacts/formal_v2/$model/best.pt" \
    --output-dir "artifacts/validation_v2/$model" --split validation --device cuda
done
"$DEMO3_PYTHON" freeze_selection.py --protocol protocol_v2.json \
  --manifest artifacts/cache_v2/manifest.json --runs artifacts/formal_v2 \
  --output artifacts/selection_v2.json
"$DEMO3_PYTHON" -u v2_pipeline.py --mode prepare-test --protocol protocol_v2.json \
  --project-root "$HIPMRI_V1_PROJECT_ROOT" --manifest artifacts/cache_v2/manifest.json \
  --raw-root /home/groups/comp3710/HipMRI_Study_open \
  --selection-freeze artifacts/selection_v2.json
for model in unet2d improved3d; do
  "$DEMO3_PYTHON" -u evaluate_v2_deterministic.py --mode evaluate --protocol protocol_v2.json \
    --execution-amendment engineering_execution_amendment.json \
    --project-root "$HIPMRI_V1_PROJECT_ROOT" --manifest artifacts/cache_v2/manifest.json \
    --model "$model" --checkpoint "artifacts/formal_v2/$model/best.pt" \
    --output-dir "artifacts/test_v2/$model" --split test --device cuda \
    --selection-freeze artifacts/selection_v2.json
done
