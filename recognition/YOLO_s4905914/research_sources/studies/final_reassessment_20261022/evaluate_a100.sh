#!/bin/bash
#SBATCH --job-name=lesion-final-eval
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=outputs/final_reassessment_%j.out
#SBATCH --error=outputs/final_reassessment_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
study=studies/final_reassessment_20261022
base_python="$PWD/.python313/bin/python"
run="$PWD/outputs/final_reassessment_20261022_${SLURM_JOB_ID}"
mkdir "$run"
"$base_python" "$study/verify_bundle.py" > "$run/bundle_verification.txt"
cp "$study/bundle_manifest.json" "$run/bundle_manifest.json"
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
cache_root="$cache_parent/comp3710-final-${UID}-${SLURM_JOB_ID}"
mkdir -m 700 "$cache_root"
mkdir "$cache_root/tmp"
export TMPDIR="$cache_root/tmp" MPLBACKEND=Agg MPLCONFIGDIR="$cache_root/matplotlib"
export YOLO_CONFIG_DIR="$cache_root/ultralytics" YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
"$base_python" -m venv "$cache_root/venv"
py="$cache_root/venv/bin/python"
printf '%s\n' "$py" > "$run/node_local_python.txt"
"$py" -m pip install --no-input --no-cache-dir -r "$study/code/requirements.txt" -c "$study/code/requirements-linux-a100.lock.txt" > "$run/install.log" 2>&1
"$py" -m pip check > "$run/pip_check.txt"
"$py" -m pip freeze > "$run/environment.freeze.txt"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > "$run/hardware.txt"
"$py" "$study/code/evaluate_locked_candidate.py" --root "$study/locked_inputs" --image-root /home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2 --output "$run/evaluation" --device 0 > "$run/evaluate.log" 2>&1
printf '%s\n' 'Completed fixed geometry seed-3710 reassessment at confidence > 0.40; no training or model selection.' > "$run/COMPLETED.txt"
