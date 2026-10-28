#!/usr/bin/env bash
# setup_wsl.sh — COMP3710 lab-computer setup (WSL), following the course guide.
#
#   1. Installs Miniconda into ~/miniconda3        (guide steps 1–3)
#   2. Accepts the Anaconda channel ToS             (guide step 4)
#   3. Creates the 'torch' env + installs PyTorch   (guide steps 5–6, plus matplotlib/pillow/numpy)
#   4. Runs conda init bash, checks CUDA            (guide section 3)
#   5. Copies your project into ~/ (Linux home is much faster than /mnt/c)
#   6. Copies ADNI AD_NC from ~/EAIT-Home/data to ~/data (local disk, faster)
#
# Every step is skipped if already done, so it is safe to re-run after a machine reset.
#
# Usage:
#   bash tools/setup_wsl.sh
#
# If you saved this file from Windows and see "$'\r': command not found", run:
#   sed -i 's/\r$//' setup_wsl.sh

set -euo pipefail

# ---------------- settings: edit these if needed ----------------
ENV_NAME="torch"
UQ_USER="s4718955"
PROJECT_SRC="/mnt/c/Users/${UQ_USER}/PatternAnalysis"   # your project on the Windows side
PROJECT_DIR="$HOME/PatternAnalysis"                     # where it will live in Linux
SRC_DATA="$HOME/EAIT-Home/data"                         # moss home, mounted on the GPU lab
DATA_DIR="$HOME/data"
# ----------------------------------------------------------------

step() { printf '\n\033[1;36m=== %s ===\033[0m\n' "$1"; }
CONDA_DIR="$HOME/miniconda3"

# ---------------- 1. Miniconda ----------------
step "1/6 Miniconda"
if [ -x "$CONDA_DIR/bin/conda" ]; then
  echo "Already installed at $CONDA_DIR"
else
  cd "$HOME"
  curl -fsSLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
  rm -f Miniconda3-latest-*.sh                # guide: tidy up the installer
fi
# conda's activate scripts reference unset variables, so relax -u while sourcing
set +u
source "$CONDA_DIR/bin/activate"
set -u

# ---------------- 2. Terms of Service ----------------
step "2/6 Anaconda channel ToS"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r    >/dev/null 2>&1 || true
echo "Accepted"

# ---------------- 3. torch environment ----------------
step "3/6 '$ENV_NAME' environment"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Environment exists"
else
  conda create -n "$ENV_NAME" python=3.11 pip -y
fi
set +u; conda activate "$ENV_NAME"; set -u

if python -c "import torch" 2>/dev/null; then
  echo "PyTorch already installed"
else
  pip3 install torch torchvision
  pip3 install matplotlib pillow numpy
fi

# Make 'conda activate torch' work in future terminals (guide note)
grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null || conda init bash >/dev/null

# ---------------- 4. CUDA check ----------------
step "4/6 CUDA check"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not found — will run on CPU"
python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Running on CPU")
EOF

# ---------------- 5. Project files ----------------
step "5/6 Project files"
if [ -f "$PROJECT_DIR/train.py" ]; then
  echo "Project already at $PROJECT_DIR"
elif [ -d "$PROJECT_SRC" ]; then
  mkdir -p "$PROJECT_DIR"
  cp -r "$PROJECT_SRC"/. "$PROJECT_DIR"/
  echo "Copied $PROJECT_SRC -> $PROJECT_DIR"
else
  echo "No project found at $PROJECT_SRC — git clone your repo into $PROJECT_DIR instead"
fi

# ---------------- 6. ADNI data ----------------
step "6/6 ADNI dataset"
ADNI_ROOT="$DATA_DIR/AD_NC"
if [ -d "$ADNI_ROOT/train" ]; then
  echo "Already present at $ADNI_ROOT"
elif [ -d "$SRC_DATA/AD_NC/train" ]; then
  mkdir -p "$DATA_DIR"
  echo "Copying $SRC_DATA/AD_NC -> $ADNI_ROOT ..."
  tar -cf - -C "$SRC_DATA" AD_NC | tar -xf - -C "$DATA_DIR"   # faster than cp for many small files
  echo "Done ($(find "$ADNI_ROOT" -type f | wc -l) files)"
else
  echo "No AD_NC found in $SRC_DATA - copy it there via moss first."
  exit 1
fi

step "Setup complete"
echo "  conda activate $ENV_NAME"
echo "  cd $PROJECT_DIR"
echo "  python train.py --root $ADNI_ROOT --epochs 30 --batch-size 64 --workers 6"