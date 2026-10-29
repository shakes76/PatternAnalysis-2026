<#
setup.ps1 - one-shot setup for COMP3710 on a Windows lab computer (no WSL needed).

  1. Installs Miniconda (per-user, no admin) if it isn't already there
  2. Accepts the Anaconda ToS, creates the 'torch' env and installs CUDA PyTorch + deps
  3. Runs conda init so 'conda activate torch' works in new PowerShell windows
  4. Checks that PyTorch can see the GPU
  5. Copies ADNI AD_NC from Rangpur (skipped if already downloaded)
  6. Prints the command to start training (does not train)

From the repo root:
    powershell -ExecutionPolicy Bypass -File .\tools\setup.ps1

Options:
    -EnvName torch                              (use comp3710 to keep an older env)
    -RangpurPath "/home/groups/comp3710/ADNI"   (folder on Rangpur that contains AD_NC)
    -DataDir "D:\data"                          (where to store ADNI locally)
    -NoData                                     (skip the ADNI download)
#>

param(
    [string]$EnvName     = "torch",
    [string]$RangpurUser = $env:USERNAME,
    [string]$RangpurHost = "rangpur.compute.eait.uq.edu.au",
    [string]$RangpurPath = "/home/groups/comp3710/ADNI",
    [string]$DataDir     = "",
    [switch]$NoData
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
$EnvDir    = "$CondaRoot\envs\$EnvName"
$Py        = "$EnvDir\python.exe"

# ------------------------------------------------ install Miniconda
Step "1/7 Miniconda"
if (Test-Path $Conda) {
    Write-Host "Already installed at $CondaRoot"
} else {
    $installer = "$env:USERPROFILE\Miniconda3-latest-Windows-x86_64.exe"
    Write-Host "Downloading installer..."
    $ProgressPreference = "SilentlyContinue"          # makes Invoke-WebRequest much faster
    Invoke-WebRequest "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $installer -ErrorAction Stop
    Write-Host "Installing (unattended, per-user, into $CondaRoot)..."
    # /S = batch install (like -b), /D = install location (like -p)
    Start-Process $installer -Wait -ErrorAction Stop -ArgumentList "/InstallationType=JustMe", "/RegisterPython=0", "/AddToPath=0", "/S", "/D=$CondaRoot"
    if (-not (Test-Path $Conda)) { throw "Miniconda install failed - check disk quota in $env:USERPROFILE" }
    Remove-Item $installer -ErrorAction SilentlyContinue   # tidy up the installer
}

# ------------------------------------------------ activate conda (this script only)
Step "2/7 Activate conda"
& "$CondaRoot\shell\condabin\conda-hook.ps1"
Write-Host "conda $(& $Conda --version)"

# ------------------------------------------------ Terms of Service
Step "3/7 Accept Anaconda channel ToS"
foreach ($ch in "main", "r", "msys2") {             # Windows defaults also include msys2
    & $Conda tos accept --override-channels --channel "https://repo.anaconda.com/pkgs/$ch" 2>$null | Out-Null
}
Write-Host "Accepted"

# ------------------------------------------------ torch env + PyTorch
Step "4/7 '$EnvName' environment + PyTorch"
if (Test-Path $Py) {
    Write-Host "Environment exists at $EnvDir"
} else {
    & $Conda create -n $EnvName python=3.11 pip -y
    Check "conda create"
}

# Probe for torch without printing a traceback
& $Py -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
    # CUDA build of PyTorch (plain 'pip install torch' on Windows is CPU-only)
    & $Py -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126
    Check "torch install"
    & $Py -m pip install --no-cache-dir matplotlib pillow numpy
    Check "pip install"
} else {
    Write-Host "PyTorch already installed"
}

# ------------------------------------------------ conda init
Step "5/7 conda init powershell"
try {
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force -ErrorAction Stop
} catch {
    Write-Host "Couldn't change execution policy (lab may lock it): $($_.Exception.Message)" -ForegroundColor Yellow
}
& $Conda init powershell | Out-Null
$policy = Get-ExecutionPolicy
if ($policy -in "Restricted", "AllSigned") {
    $CondaInitOk = $false
    Write-Host "Execution policy is '$policy', so PowerShell won't load conda at startup." -ForegroundColor Yellow
} else {
    $CondaInitOk = $true
    Write-Host "Done - new PowerShell windows will start in (base)"
}

# ------------------------------------------------ does CUDA work?
Step "6/7 CUDA check"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
} else {
    Write-Host "nvidia-smi not found - will run on CPU" -ForegroundColor Yellow
}
& $Py -c "import torch; ok = torch.cuda.is_available(); print('CUDA available:', ok); print('Device name:', torch.cuda.get_device_name(0) if ok else 'Running on CPU')"

# ------------------------------------------------ ADNI data
$AdniRoot = Join-Path $DataDir "AD_NC"
if ($NoData) {
    Step "7/7 ADNI dataset (skipped)"
} else {
    Step "7/7 ADNI dataset"
    if (Test-Path (Join-Path $AdniRoot "train")) {
        Write-Host "Already present at $AdniRoot"
    } else {
        New-Item -ItemType Directory -Force -Path $DataDir | Out-Null
        $remote = "$RangpurUser@$RangpurHost"
        $tar    = Join-Path $DataDir "AD_NC.tar"
        Write-Host "Packing AD_NC on Rangpur (UQ network/VPN needed; password asked up to 3 times)..."
        # One tar file copies far faster than thousands of individual JPEGs
        ssh $remote "tar -cf ~/AD_NC.tar -C $RangpurPath AD_NC"
        Check "remote tar (is -RangpurPath right? try: ssh $remote ls $RangpurPath)"
        Write-Host "Downloading..."
        scp "${remote}:~/AD_NC.tar" $tar
        Check "scp"
        ssh $remote "rm -f ~/AD_NC.tar"                # tidy up your Rangpur home quota
        Write-Host "Extracting..."
        tar -xf $tar -C $DataDir                       # tar.exe ships with Windows 10/11
        Check "extract"
        Remove-Item $tar
        if (-not (Test-Path (Join-Path $AdniRoot "train"))) { throw "Extracted, but $AdniRoot\train not found - check folder layout" }
    }
}

# ------------------------------------------------ Done
Step "Setup complete"
if ($CondaInitOk) {
    Write-Host "Close this window, open a NEW PowerShell, then:"
} else {
    Write-Host "Open 'Anaconda PowerShell Prompt' from the Start menu (or run the hook line first), then:"
    Write-Host "  & $CondaRoot\shell\condabin\conda-hook.ps1"
}
Write-Host "  conda activate $EnvName" -ForegroundColor Green
Write-Host "  cd `"$ProjectDir`"" -ForegroundColor Green
Write-Host "  python train.py --root data\AD_NC --epochs 30 --batch-size 64 --workers 6" -ForegroundColor Green