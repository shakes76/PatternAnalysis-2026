#!/bin/bash
# One locked supplementary evaluation per model; no training or threshold search.
#SBATCH --job-name=lesion-reassessment-ns
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=outputs/reassessment_%j.out
#SBATCH --error=outputs/reassessment_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
code=recognition/YOLO_s4905914
config="$code/experiment_provisional.json"
base_python="$PWD/.python313/bin/python"
training=outputs/provisional_20261017_669143
run="outputs/reassessment_20261017_${SLURM_JOB_ID}"
mkdir "$run"
"$base_python" - <<'PY'
from pathlib import Path
import hashlib, json
code=Path('recognition/YOLO_s4905914'); original=Path('outputs/provisional_20261017_669143')
plan=json.loads((code/'reassessment_plan.json').read_text())
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(code/'experiment_provisional.json')==plan['config_sha256']
assert (original/'COMPLETED.txt').is_file()
for model in ('n','s'):
    expected=plan['models'][model]
    lock=original/f'val_{model}/selection_lock.json'
    assert sha(lock)==expected['selection_lock_sha256']
    assert json.loads(lock.read_text())['threshold']==expected['threshold']
    assert sha(original/f'train_{model}/fit/weights/best.pt')==expected['checkpoint_sha256']
    training=json.loads((original/f'train_{model}/run.json').read_text())
    for name in ('train.py','evaluation.py','dataset.py','runtime.py','modules.py','preprocessing.py'):
        assert sha(code/name)==training['source_sha256'][name],name
PY
cache_parent=$("$base_python" - <<'PY'
import os, shutil
from pathlib import Path
for path in (os.environ.get('SLURM_TMPDIR'),os.environ.get('TMPDIR'),'/cache','/scratch','/tmp'):
    if path and Path(path).is_dir() and os.access(path,os.W_OK|os.X_OK) and shutil.disk_usage(path).free>=12*1024**3:
        print(path)
        break
else:
    raise RuntimeError('No writable temporary directory with 12 GiB free')
PY
)
cache_root="$cache_parent/comp3710-reassessment-${UID}-${SLURM_JOB_ID}"
mkdir -m 700 "$cache_root"
mkdir "$cache_root/tmp"
export TMPDIR="$cache_root/tmp" MPLBACKEND=Agg MPLCONFIGDIR="$cache_root/matplotlib"
export YOLO_CONFIG_DIR="$cache_root/ultralytics" YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
"$base_python" -m venv "$cache_root/venv"
py="$cache_root/venv/bin/python"
"$py" -m pip install --no-input --no-cache-dir -r "$code/requirements.txt" \
  -c "$code/requirements-linux-a100.lock.txt" > "$run/install.log" 2>&1
"$py" -m pip check > "$run/pip_check.txt"
"$py" -m pip freeze > "$run/environment.freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.txt"
cp "$config" "$run/locked_config.json"
cp "$code/reassessment_plan.json" "$run/reassessment_plan.json"
stage="$training/inputs/staged"
for model in n s; do
  checkpoint="$training/train_$model/fit/weights/best.pt"
  lock="$training/val_$model/selection_lock.json"
  cp "$lock" "$run/selection_lock_$model.json"
  "$py" "$code/train.py" evaluate --config "$config" --prepared data/known_lesion_v1 \
    --stage "$stage" --checkpoint "$checkpoint" --split reassessment --lock "$lock" \
    --output "$run/eval_$model" --device 0 > "$run/eval_$model.log" 2>&1
  "$py" "$code/benchmark_inference.py" --config "$config" --checkpoint "$checkpoint" \
    --lock "$lock" --manifest data/known_lesion_v1/val.csv --image-root "$stage/val/images" \
    --output "$run/benchmark_$model" --device 0 > "$run/benchmark_$model.log" 2>&1
done
printf '%s\n' 'Locked supplementary n/s evaluation and validation-only resource benchmarking completed; no retraining or threshold selection.' > "$run/COMPLETED.txt"
