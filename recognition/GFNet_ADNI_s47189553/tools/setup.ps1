<#
setup.ps1 - one-shot setup for COMP3710 ADNI on a Windows + NVIDIA machine.

  1. Installs Miniconda (per-user, no admin) if it isn't already there
  2. Creates the conda env and installs CUDA PyTorch + deps
  3. Copies ADNI AD_NC from Rangpur (skipped if already downloaded)
  4. Prints the command to start training (does not train)

Keep it in the repo's tools\ folder (or next to train.py). From the repo root:
    powershell -ExecutionPolicy Bypass -File .\tools\setup.ps1

Options:
    -RangpurPath "/home/groups/comp3710/ADNI"   (folder on Rangpur that contains AD_NC)
    -DataDir "D:\data"    (where to store ADNI locally)
#>

param(
    [string]$EnvName     = "comp3710",
    [string]$RangpurUser = $env:USERNAME,
    [string]$RangpurHost = "rangpur.compute.eng.uq.edu.au",
    [string]$RangpurPath = "/home/groups/comp3710/ADNI",
    [string]$DataDir     = ""
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns any stderr line from a native
# program (pip warnings, conda progress) into a fatal error under "Stop".
# Native commands are checked with Check (exit code) instead.
$ErrorActionPreference = "Continue"
function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit code $LASTEXITCODE)" } }

# Works whether this script sits next to train.py or in a tools\ subfolder
$ProjectDir = if (Test-Path "$PSScriptRoot\train.py") { $PSScriptRoot } else { Split-Path $PSScriptRoot -Parent }
if (-not (Test-Path "$ProjectDir\train.py")) { throw "Can't find train.py in $PSScriptRoot or its parent folder" }
if (-not $DataDir) { $DataDir = "$ProjectDir\data" }

$CondaRoot = "$env:USERPROFILE\miniconda3"
$Conda     = "$CondaRoot\Scripts\conda.exe"
$Py        = "$CondaRoot\envs\$EnvName\python.exe"

# ---------------------------------------------------------------- 1. Miniconda
Step "1/4 Miniconda"
if (Test-Path $Conda) {
    Write-Host "Already installed at $CondaRoot"
} else {
    $installer = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
    Write-Host "Downloading installer..."
    $ProgressPreference = "SilentlyContinue"          # makes Invoke-WebRequest much faster
    Invoke-WebRequest "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $installer -ErrorAction Stop
    Write-Host "Installing (silent, per-user)..."
    Start-Process $installer -Wait -ErrorAction Stop -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$CondaRoot"
    if (-not (Test-Path $Conda)) { throw "Miniconda install failed - check disk quota in $env:USERPROFILE" }
}

# ---------------------------------------------------------------- 2. Python env
Step "2/4 Python environment '$EnvName'"
if (-not (Test-Path $Py)) {
    # conda-forge only, so no Anaconda Terms-of-Service prompt blocks the script
    & $Conda create -y -n $EnvName --override-channels -c conda-forge python=3.11
    Check "conda create"
}

# Probe for torch without printing a traceback
& $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    & $Py -m pip install --upgrade pip
    & $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    Check "torch install"
    & $Py -m pip install matplotlib pillow numpy
    Check "pip install"
} else {
    Write-Host "PyTorch already installed"
}

& $Py -c "import torch; ok=torch.cuda.is_available(); print('CUDA:', ok, '|', torch.cuda.get_device_name(0) if ok else 'NO GPU'); exit(0 if ok else 1)"
Check "CUDA check (PyTorch can't see the GPU)"

# ---------------------------------------------------------------- 3. ADNI data
Step "3/4 ADNI dataset"
$AdniRoot = Join-Path $DataDir "AD_NC"
if (Test-Path (Join-Path $AdniRoot "train")) {
    Write-Host "Already present at $AdniRoot"
} else {
    New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
    $remote = "$RangpurUser@$RangpurHost"
    $tar    = Join-Path $DataDir "AD_NC.tar"
    Write-Host "Packing AD_NC on Rangpur (you'll be asked for your UQ password up to 3 times)..."
    # One tar file copies far faster than thousands of individual JPEGs
    ssh $remote "tar -cf ~/AD_NC.tar -C $RangpurPath AD_NC"
    Check "remote tar (is -RangpurPath right? try: ssh $remote ls $RangpurPath)"
    Write-Host "Downloading..."
    scp "${remote}:~/AD_NC.tar" $tar
    Check "scp"
    ssh $remote "rm -f ~/AD_NC.tar"                    # tidy up your Rangpur home quota
    Write-Host "Extracting..."
    tar -xf $tar -C $DataDir                           # tar.exe ships with Windows 10/11
    Check "extract"
    Remove-Item $tar
    if (-not (Test-Path (Join-Path $AdniRoot "train"))) { throw "Extracted, but $AdniRoot\train not found - check folder layout" }
}

# ---------------------------------------------------------------- 4. Done
Step "4/4 Setup complete"
Write-Host "To train, run from $ProjectDir :"
Write-Host "  cd `"$ProjectDir`""
Write-Host "  & `"$Py`" train.py --root `"$AdniRoot`" --epochs 30 --batch-size 64 --workers 6" -ForegroundColor Green