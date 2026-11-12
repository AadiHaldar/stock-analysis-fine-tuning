# PowerShell script to set up virtual environment and install dependencies
# Run this from the project root directory.

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setting up local Stock-Analysis AI system..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in system PATH." -ForegroundColor Red
    Exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path ".venv")) {
    Write-Host "Creating virtual environment '.venv'..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create virtual environment." -ForegroundColor Red
        Exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment '.venv' already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$venvPath = ".venv\Scripts\Activate.ps1"
if (Test-Path -Path $venvPath) {
    . $venvPath
} else {
    Write-Host "Failed to find activation script at $venvPath." -ForegroundColor Red
    Exit 1
}

# Upgrade pip
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to upgrade pip." -ForegroundColor Red
    Exit 1
}

# Install PyTorch with CUDA 12.8 support (required for RTX 5060 sm_120)
Write-Host "Installing PyTorch 2.7+ with CUDA 12.8 support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install PyTorch." -ForegroundColor Red
    Exit 1
}
Write-Host "PyTorch installed successfully." -ForegroundColor Green

# Install remaining dependencies from requirements.txt
Write-Host "Installing packages from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install dependencies from requirements.txt." -ForegroundColor Red
    Exit 1
}
Write-Host "Requirements installed successfully." -ForegroundColor Green

# (Optional) Install Unsloth for Windows if needed
# Write-Host "Installing Unsloth for Windows (for fast QLoRA training)..." -ForegroundColor Yellow
# pip install "unsloth[windows] @ git+https://github.com/unslothai/unsloth.git"
# if ($LASTEXITCODE -ne 0) {
#     Write-Host "Unsloth installation failed. We will fall back to default HF PEFT + bitsandbytes if needed." -ForegroundColor Yellow
# } else {
#     Write-Host "Unsloth installed successfully." -ForegroundColor Green
# }

# Verify CUDA detection
Write-Host "Verifying GPU installation..." -ForegroundColor Yellow
python -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('CUDA Device Capability:', torch.cuda.get_device_capability(0))
"

Write-Host "=========================================" -ForegroundColor Green
Write-Host "Setup Completed! You can now run the scripts." -ForegroundColor Green
Write-Host "Make sure to run '.venv\Scripts\Activate.ps1' in your shell sessions." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
