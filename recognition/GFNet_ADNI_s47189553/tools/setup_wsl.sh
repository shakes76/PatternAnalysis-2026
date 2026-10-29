#!/usr/bin/env bash
# setup_wsl.sh — COMP3710 lab-computer setup (WSL), following the course guide.
#
#   1. Installs Miniconda into ~/miniconda3        (guide steps 1–3)
#   2. Accepts the Anaconda channel ToS             (guide step 4)
#   3. Creates the 'torch' env + installs PyTorch   (guide steps 5–6, plus matplotlib/pillow/numpy)
#   4. Runs conda init bash, checks CUDA            (guide section 3)
#   5. Copies your project into ~/ (Linux home is much faster than /mnt/c)
#   6. Puts ADNI at ~/PatternAnalysis/data/AD_NC (skipped if already there),
#      extracting ~/EAIT-Home/data/AD_NC.tar, or copying the AD_NC folder if no tar
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
DATA_DIR="$PROJECT_DIR/data"                            # ADNI ends up at $DATA_DIR/AD_NC
TORCH_INDEX="https://download.pytorch.org/whl/cu126"    # pinned CUDA 12.6 build of PyTorch
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

# A system/other cuDNN on LD_LIBRARY_PATH can override PyTorch's own copy
unset LD_LIBRARY_PATH

install_torch() {
  pip3 install --force-reinstall --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
  pip3 install --no-cache-dir matplotlib pillow numpy
}

if python -c "import torch" 2>/dev/null; then
  echo "PyTorch already installed"
else
  install_torch
fi

# Make 'conda activate torch' work in future terminals (guide note)
grep -q "conda initialize" "$HOME/.bashrc" 2>/dev/null || conda init bash >/dev/null

# ---------------- 4. CUDA check ----------------
step "4/6 CUDA check"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "nvidia-smi not found — will run on CPU"
gpu_ok() {
  python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Running on CPU")
if torch.cuda.is_available():
    # A real conv loads cuDNN; is_available() alone does not
    y = torch.nn.Conv2d(1, 8, 3).cuda()(torch.randn(2, 1, 224, 224, device="cuda"))
    torch.cuda.synchronize()
    print("Conv test OK:", tuple(y.shape))
EOF
}

if ! gpu_ok; then
  echo "GPU test failed - reinstalling PyTorch from $TORCH_INDEX ..."
  install_torch
  gpu_ok || { echo "Still failing. Check free space with: df -h ~"; exit 1; }
fi

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
if [ -d "$ADNI_ROOT/train/AD" ] && [ -d "$ADNI_ROOT/train/NC" ] && [ -d "$ADNI_ROOT/test" ]; then
  echo "ADNI already exists at $ADNI_ROOT ($(find "$ADNI_ROOT" -type f | wc -l) files) - skipping"
elif [ -f "$SRC_DATA/AD_NC.tar" ]; then
  # One big file reads from the network drive much faster than ~30k small JPEGs
  mkdir -p "$DATA_DIR"
  echo "Extracting $SRC_DATA/AD_NC.tar -> $DATA_DIR ..."
  tar -xf "$SRC_DATA/AD_NC.tar" -C "$DATA_DIR"
  [ -d "$ADNI_ROOT/train" ] || { echo "Extracted, but $ADNI_ROOT/train is missing - check the tar's folder layout"; exit 1; }
  echo "Done ($(find "$ADNI_ROOT" -type f | wc -l) files)"
elif [ -d "$SRC_DATA/AD_NC/train" ]; then
  mkdir -p "$DATA_DIR"
  echo "Copying $SRC_DATA/AD_NC -> $ADNI_ROOT ..."
  tar -cf - -C "$SRC_DATA" AD_NC | tar -xf - -C "$DATA_DIR"   # faster than cp for many small files
  echo "Done ($(find "$ADNI_ROOT" -type f | wc -l) files)"
else
  echo "No ADNI at $ADNI_ROOT, and no AD_NC.tar or AD_NC folder in $SRC_DATA - copy it there via moss first."
  exit 1
fi

step "Setup complete"
echo "  conda activate $ENV_NAME"
echo "  cd $PROJECT_DIR"
echo "  python train.py --root data/AD_NC --epochs 30 --batch-size 64 --workers 6"