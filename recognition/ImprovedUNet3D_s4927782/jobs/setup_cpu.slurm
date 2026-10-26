#!/bin/bash
#SBATCH --job-name=hipmri-env
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
# This cluster advertises RealMemory=1 MiB per node with an unlimited default.
# An explicit GiB request is rejected; use its configured memory default.
cd "$SLURM_SUBMIT_DIR"
export TMPDIR="${SLURM_TMPDIR:-$SLURM_SUBMIT_DIR/artifacts/tmp}"
mkdir -p "$TMPDIR"

# Reuse torch read-only; install the new task's extras in its own environment.
base_python="$HOME/miniconda3/envs/torch/bin/python"
"$base_python" -m venv --system-site-packages .venv
.venv/bin/python -m pip install --no-cache-dir -r requirements-extra.txt
.venv/bin/python - <<'PY'
from importlib.metadata import version
import json
import platform
print(json.dumps({"python": platform.python_version(), "packages": {
    name: version(name) for name in ["torch", "numpy", "nibabel", "scipy", "matplotlib"]
}}, indent=2))
PY
