#!/bin/bash
#SBATCH --job-name=lesion-known-opt
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=outputs/known_opt_%j.out
#SBATCH --error=outputs/known_opt_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
study=studies/known_lesion_optimization_20261021
code="$study/code"
base_python="$PWD/.python313/bin/python"
run="outputs/known_lesion_optimization_20261021_${SLURM_JOB_ID}"
mkdir "$run"
"$base_python" "$study/verify_preparation.py" > "$run/preparation_check.log" 2>&1
cp "$study/plan.json" "$run/plan.json"
cp "$study/verification.json" "$run/verification.json"
cache_parent=$("$base_python" - <<'PY'
import os,shutil
from pathlib import Path
for p in (os.environ.get('SLURM_TMPDIR'),os.environ.get('TMPDIR'),'/cache','/scratch','/tmp'):
    if p and Path(p).is_dir() and os.access(p,os.W_OK|os.X_OK) and shutil.disk_usage(p).free>=12*1024**3:
        print(p)
        break
else: raise RuntimeError('No writable scratch directory with 12 GiB free')
PY
)
cache_root="$cache_parent/comp3710-known-opt-${UID}-${SLURM_JOB_ID}"
mkdir -m 700 "$cache_root"
mkdir "$cache_root/tmp"
export TMPDIR="$cache_root/tmp" MPLBACKEND=Agg MPLCONFIGDIR="$cache_root/matplotlib"
export YOLO_CONFIG_DIR="$cache_root/ultralytics" YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
"$base_python" -m venv "$cache_root/venv"
py="$cache_root/venv/bin/python"
printf '%s\n' "$py" > "$run/node_local_python.txt"
"$py" -m pip install --no-input --no-cache-dir -r "$code/requirements.txt" -c "$code/requirements-linux-a100.lock.txt" > "$run/install.log" 2>&1
"$py" -m pip check > "$run/pip_check.txt"
"$py" -m pip freeze > "$run/environment.freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.txt"
variants=(n_cls1 n_geometry n_long120)
variant=${variants[$SLURM_ARRAY_TASK_ID]}
for variant in "$variant"; do
  config="$study/$variant.json"
  out="$run/$variant"
  mkdir "$out"
  cp "$config" "$out/locked_config.json"
  "$py" "$code/audit_real_inputs.py" --config "$config" --prepared data/known_lesion_v1 --sources sources.json --output "$out/inputs" > "$out/input_audit.log" 2>&1
  stage="$out/inputs/staged"
  "$py" "$code/train.py" fit --config "$config" --prepared data/known_lesion_v1 --stage "$stage" --weights-dir weights/original_coco --model n --output "$out/train_n" --device 0 > "$out/train_n.log" 2>&1
  "$py" "$code/train.py" evaluate --config "$config" --prepared data/known_lesion_v1 --stage "$stage" --checkpoint "$out/train_n/fit/weights/best.pt" --split val --output "$out/val_n" --device 0 > "$out/val_n.log" 2>&1
  printf '%s\n' "${variant} completed: train/validation only" >> "$run/progress.txt"
done
printf '%s\n' 'Assigned known-lesion ablation completed; validation only; no reassessment inference.' > "$run/COMPLETED.txt"
