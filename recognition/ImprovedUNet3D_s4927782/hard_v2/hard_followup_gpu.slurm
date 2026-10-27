#!/bin/bash
#SBATCH --job-name=hipmri-hard-only-followup
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/artifacts/pycache"
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/artifacts/matplotlib"
DEMO3_V1_PROJECT=/home/Student/s4927782/comp3710-demo3
DEMO3_PYTHON="$DEMO3_V1_PROJECT/.venv/bin/python"
mkdir artifacts/hard_followup_execution.lock
common=(--protocol protocol_v2.json --scope-plan hard_followup_protocol.json \
  --project-root "$DEMO3_V1_PROJECT" --manifest artifacts/cache_v2/manifest.json)
"$DEMO3_PYTHON" -u train_hard_followup.py --mode train "${common[@]}" \
  --qa-receipt artifacts/qa_v2/qa_receipt.json --device cuda \
  --output-dir artifacts/hard_followup/improved3d
"$DEMO3_PYTHON" -u train_hard_followup.py --mode evaluate "${common[@]}" \
  --checkpoint artifacts/hard_followup/improved3d/best.pt --device cuda \
  --split validation --output-dir artifacts/hard_validation/improved3d
"$DEMO3_PYTHON" -u train_hard_followup.py --mode freeze "${common[@]}" \
  --run-dir artifacts/hard_followup/improved3d --selection-freeze artifacts/hard_selection.json
"$DEMO3_PYTHON" -u train_hard_followup.py --mode prepare-test "${common[@]}" \
  --selection-freeze artifacts/hard_selection.json \
  --raw-root /home/groups/comp3710/HipMRI_Study_open
"$DEMO3_PYTHON" -u train_hard_followup.py --mode evaluate "${common[@]}" \
  --checkpoint artifacts/hard_followup/improved3d/best.pt --device cuda \
  --selection-freeze artifacts/hard_selection.json --split test \
  --output-dir artifacts/hard_test/improved3d
