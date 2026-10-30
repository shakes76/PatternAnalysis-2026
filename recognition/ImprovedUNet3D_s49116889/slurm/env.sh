# Shared job setup: activate a conda env on the compute node itself (not
# inherited from the login shell) and make sure PyTorch can use the GPU.
#   1st choice: `torch`    (PyTorch 2.14, CUDA 13 build - needs a recent driver)
#   fallback:   `torch118` (PyTorch 2.7.1, CUDA 11.8 build - works on older drivers)
CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"
source "$CONDA_BASE/etc/profile.d/conda.sh"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

cuda_ok() { python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; }

conda activate torch
if [ -n "$SLURM_JOB_ID" ] && ! cuda_ok; then
    echo "env 'torch' cannot use the GPU on $(hostname); trying fallback env 'torch118'"
    conda activate torch118
fi
echo "python: $(which python)"
python -c "import matplotlib" 2>/dev/null || python -m pip install --quiet matplotlib
python -c "import torch, nibabel, matplotlib; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# Inside a Slurm job, stop immediately if PyTorch still cannot use the GPU,
# rather than silently training on the CPU for hours.
if [ -n "$SLURM_JOB_ID" ] && ! cuda_ok; then
    echo "ERROR: CUDA not available to PyTorch on $(hostname) - stopping job."
    exit 1
fi
