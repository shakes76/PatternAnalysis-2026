#!/bin/bash
#SBATCH --job-name=lesion-final-speed
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=outputs/final_benchmark_%j.out
#SBATCH --error=outputs/final_benchmark_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch}"
cd "$SLURM_SUBMIT_DIR"
study=studies/final_benchmark_20261028
code=recognition/YOLO_s4905914
base_python="$PWD/.python313/bin/python"
run="$PWD/outputs/final_benchmark_20261028_${SLURM_JOB_ID}"
mkdir "$run"
sha256sum --check "$study/inputs.sha256" > "$run/inputs_before.txt"
cp "$study/inputs.sha256" "$study/benchmark_a100.sh" "$run/"
hostname > "$run/hostname.txt"
date -u +%FT%TZ > "$run/started_utc.txt"
cache_parent=$("$base_python" - <<'PY'
import os, shutil
from pathlib import Path
for p in (os.environ.get('SLURM_TMPDIR'), os.environ.get('TMPDIR'), '/cache', '/scratch', '/tmp'):
    if p and Path(p).is_dir() and os.access(p, os.W_OK | os.X_OK) and shutil.disk_usage(p).free >= 12*1024**3:
        print(p)
        break
else:
    raise RuntimeError('No writable scratch directory with 12 GiB free')
PY
)
cache_root="$cache_parent/comp3710-final-benchmark-${UID}-${SLURM_JOB_ID}"
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
"$py" "$code/benchmark_inference.py" \
  --config studies/known_lesion_optimization_20261021/n_geometry.json \
  --checkpoint outputs/known_lesion_optimization_20261021_681452/n_geometry/train_n/fit/weights/best.pt \
  --lock outputs/known_lesion_optimization_20261021_681452/n_geometry/val_n/selection_lock.json \
  --manifest data/known_lesion_v1/val.csv \
  --image-root outputs/provisional_20261017_669143/inputs/staged/val/images \
  --output "$run/benchmark_geometry_n" --device 0 > "$run/benchmark.log" 2>&1
sha256sum --check "$study/inputs.sha256" > "$run/inputs_after.txt"
date -u +%FT%TZ > "$run/completed_utc.txt"
printf '%s\n' 'Completed validation-only resource timing. Fixed geometry n checkpoint and confidence > 0.40; no training, accuracy evaluation or model selection.' > "$run/COMPLETED.txt"
