#!/bin/bash
# Exploratory n-only expanded training; identical held-out sets, no reassessment inference.
# Missing identity links are unresolved; this is not an independent test claim.
#SBATCH --job-name=lesion-expanded-n
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/expanded_%j.out
#SBATCH --error=outputs/expanded_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
study=studies/expanded_training_20261017
code="$study/code"
config="$study/experiment_expanded.json"
prepared=data/expanded_training_v1
base_python="$PWD/.python313/bin/python"
run="outputs/expanded_training_20261017_${SLURM_JOB_ID}"
mkdir "$run"
"$base_python" - <<'PY'
from pathlib import Path
import hashlib,json
study=Path('studies/expanded_training_20261017')
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
prep=json.loads((study/'preparation.json').read_text())
cfg=json.loads((study/'experiment_expanded.json').read_text())
base=Path('recognition/YOLO_s4905914/experiment_provisional.json')
assert sha(base)==prep['baseline_config_sha256']
assert sha(study/'experiment_expanded.json')==prep['config_sha256']
assert cfg['purpose']=='exploratory_expanded_training'
assert cfg['ready_for_full_training'] and cfg['status']=='locked_before_training'
assert not cfg['course_acceptance_confirmed']
original=json.loads(base.read_text())
for key in ('train','evaluation','models','classes','checkpoint_selection'):
    assert cfg[key]==original[key], key
for name,digest in prep['snapshot_runtime_source_sha256'].items():
    assert sha(study/'code'/name)==digest,name
assert prep['counts']['train']=={'images':2263,'Benign':1826,'Malignant':437}
assert prep['held_out_manifests_and_labels_unchanged']
assert not prep['reassessment_inference_planned']
assert json.loads((study/'verification.json').read_text())['status']=='passed'
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
cache_root="$cache_parent/comp3710-expanded-${UID}-${SLURM_JOB_ID}"
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
cp "$study/preparation.json" "$run/preparation.json"
cp "$study/verification.json" "$run/preparation_verification.json"
"$py" "$code/train.py" plan --config "$config" --prepared "$prepared" > "$run/plan.json"
"$py" "$code/preflight.py" --config "$config" --weights-dir weights/original_coco \
  --output "$run/gpu_preflight" --device 0 > "$run/preflight.log" 2>&1
"$py" "$code/audit_real_inputs.py" --config "$config" --prepared "$prepared" \
  --sources sources.json --output "$run/inputs" > "$run/input_audit.log" 2>&1
stage="$run/inputs/staged"
for model in n; do
  "$py" "$code/train.py" fit --config "$config" --prepared "$prepared" \
    --stage "$stage" --weights-dir weights/original_coco --model "$model" \
    --output "$run/train_$model" --device 0 > "$run/train_$model.log" 2>&1
  "$py" "$code/train.py" evaluate --config "$config" --prepared "$prepared" \
    --stage "$stage" --checkpoint "$run/train_$model/fit/weights/best.pt" \
    --split val --output "$run/val_$model" --device 0 > "$run/val_$model.log" 2>&1
done
printf '%s\n' 'Exploratory expanded n training and validation completed; no reassessment inference; unknown identity leakage possible.' > "$run/COMPLETED.txt"
