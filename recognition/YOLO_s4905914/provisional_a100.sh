#!/bin/bash
# Fixed n/s candidate study: train and validation only, no reassessment.
# Instructor acceptance remains unconfirmed in experiment_provisional.json.
#SBATCH --job-name=lesion-provisional-ns
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/provisional_%j.out
#SBATCH --error=outputs/provisional_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
code=recognition/YOLO_s4905914
config="$code/experiment_provisional.json"
base_python="$PWD/.python313/bin/python"
run="outputs/provisional_20261017_${SLURM_JOB_ID}"
mkdir "$run"
"$base_python" - <<'PY'
from pathlib import Path
import hashlib, json
code=Path('recognition/YOLO_s4905914')
cfg=json.loads((code/'experiment_provisional.json').read_text())
base=json.loads((code/'experiment.json').read_text())
assert cfg['ready_for_full_training'] and cfg['status']=='locked_before_training'
assert cfg['course_acceptance_confirmed'] is False
for field in ('train','evaluation','models','classes','manifest_sha256',
              'split_class_counts','model_policy','checkpoint_selection'):
    assert cfg[field]==base[field], f'Unreviewed change: {field}'
evidence=Path('outputs/school_readiness_669120')
assert (evidence/'COMPLETED.txt').is_file()
gpu=evidence/'gpu_preflight/preflight.json'
assert hashlib.sha256(gpu.read_bytes()).hexdigest()==cfg['release_decision']['readiness_gpu_sha256']
assert hashlib.sha256((code/'experiment.json').read_bytes()).hexdigest()==cfg['release_decision']['draft_config_sha256']
assert json.loads(gpu.read_text())['status']=='completed'
PY
# Rangpur supplies a writable per-job scratch directory through TMPDIR.
cache_parent=$("$base_python" - <<'PY'
import os, shutil
from pathlib import Path
for p in (os.environ.get('SLURM_TMPDIR'),os.environ.get('TMPDIR'),'/cache','/scratch','/tmp'):
    if p and Path(p).is_dir() and os.access(p,os.W_OK|os.X_OK) and shutil.disk_usage(p).free>=12*1024**3:
        print(p)
        break
else:
    raise RuntimeError('No writable temporary directory with 12 GiB free')
PY
)
cache_root="$cache_parent/comp3710-provisional-${UID}-${SLURM_JOB_ID}"
mkdir -m 700 "$cache_root"
mkdir "$cache_root/tmp"
export TMPDIR="$cache_root/tmp" MPLBACKEND=Agg MPLCONFIGDIR="$cache_root/matplotlib"
export YOLO_CONFIG_DIR="$cache_root/ultralytics" YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
"$base_python" -m venv "$cache_root/venv"
py="$cache_root/venv/bin/python"
printf '%s\n' "$py" > "$run/node_local_python.txt"
"$py" -m pip install --no-input --no-cache-dir -r "$code/requirements.txt" \
  -c "$code/requirements-linux-a100.lock.txt" > "$run/install.log" 2>&1
"$py" -m pip check > "$run/pip_check.txt"
"$py" -m pip freeze > "$run/environment.freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.txt"
cp "$config" "$run/locked_config.json"
"$py" "$code/train.py" plan --config "$config" --prepared data/known_lesion_v1 > "$run/plan.json"
"$py" "$code/preflight.py" --config "$config" --weights-dir weights/original_coco \
  --output "$run/gpu_preflight" --device 0 > "$run/preflight.log" 2>&1
"$py" "$code/audit_real_inputs.py" --config "$config" --prepared data/known_lesion_v1 \
  --sources sources.json --output "$run/inputs" > "$run/input_audit.log" 2>&1
stage="$run/inputs/staged"
for model in n s; do
  "$py" "$code/train.py" fit --config "$config" --prepared data/known_lesion_v1 \
    --stage "$stage" --weights-dir weights/original_coco --model "$model" \
    --output "$run/train_$model" --device 0 > "$run/train_$model.log" 2>&1
  "$py" "$code/train.py" evaluate --config "$config" --prepared data/known_lesion_v1 \
    --stage "$stage" --checkpoint "$run/train_$model/fit/weights/best.pt" \
    --split val --output "$run/val_$model" --device 0 > "$run/val_$model.log" 2>&1
done
printf '%s\n' 'Provisional n/s training and validation completed; reassessment not evaluated; instructor acceptance unconfirmed.' > "$run/COMPLETED.txt"
