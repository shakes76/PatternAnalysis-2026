#!/bin/bash
# A bounded preparation job. No real-image model training or inference occurs.
# Submit from the restart project root after provisioning .python313 separately.
# This environment is node-local and disposable; a later job must recheck it.
#SBATCH --job-name=lesion-restart-readiness
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=outputs/readiness_%j.out
#SBATCH --error=outputs/readiness_%j.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with sbatch; do not run on a login node}"
cd "$SLURM_SUBMIT_DIR"
code=recognition/YOLO_s4905914
base_python="$PWD/.python313/bin/python"
test -x "$base_python"
run="outputs/school_readiness_${SLURM_JOB_ID}"
mkdir "$run"
cache_parent=$("$base_python" - <<'PY'
import json, os, shutil, sys
from pathlib import Path
assert sys.version_info[:2] == (3, 13), 'Use the tested Python 3.13 family'
cfg=json.loads(Path('recognition/YOLO_s4905914/experiment.json').read_text())
assert cfg['ready_for_full_training'] is False, 'This job only verifies the preparation draft'
for candidate in (os.environ.get('SLURM_TMPDIR'), os.environ.get('TMPDIR'),
                  '/cache', '/scratch', '/tmp'):
    if (candidate and Path(candidate).is_dir() and os.access(candidate, os.W_OK | os.X_OK)
            and shutil.disk_usage(candidate).free >= 12 * 1024**3):
        print(candidate)
        break
else:
    raise RuntimeError('No writable node-local temporary directory with 12 GiB free')
PY
)
cache_root="$cache_parent/comp3710-restart-${UID}-${SLURM_JOB_ID}"
test ! -e "$cache_root"
mkdir -m 700 "$cache_root"
mkdir "$cache_root/tmp"
export TMPDIR="$cache_root/tmp"
export MPLBACKEND=Agg MPLCONFIGDIR="$cache_root/matplotlib"
export YOLO_CONFIG_DIR="$cache_root/ultralytics" YOLO_OFFLINE=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
printf '%s\n' "$cache_root/venv" > "$run/node_local_environment.txt"
"$base_python" -m venv "$cache_root/venv"
runtime_python="$cache_root/venv/bin/python"
"$runtime_python" -m pip install --no-input --no-cache-dir -r "$code/requirements.txt" > "$run/install.log" 2>&1
"$runtime_python" -m pip check > "$run/pip_check.txt"
"$runtime_python" -m pip freeze > "$run/environment.freeze.txt"
"$runtime_python" -c "import sys; sys.path.insert(0, '$code'); from runtime import check_runtime; print(check_runtime())" > "$run/runtime.txt"
"$runtime_python" -m unittest discover -s "$code" -p 'test_*.py' -v > "$run/unit_tests.log" 2>&1
"$runtime_python" "$code/audit_real_inputs.py" --config "$code/experiment.json" \
  --prepared data/known_lesion_v1 --sources sources.json --output "$run/real_inputs"
"$runtime_python" "$code/preflight.py" --config "$code/experiment.json" \
  --weights-dir weights/original_coco --output "$run/gpu_preflight" --device 0
"$runtime_python" "$code/smoke_test.py" --base-config "$code/experiment.json" \
  --weights-dir weights/original_coco --output "$run/synthetic_cpu" --warmup-epochs 3 \
  > "$run/synthetic_cpu.log" 2>&1
printf '%s\n' 'Readiness checks completed; formal ISIC training remains disabled.' > "$run/COMPLETED.txt"
