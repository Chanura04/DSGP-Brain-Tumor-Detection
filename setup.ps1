# Exit on error
$ErrorActionPreference = "Stop"

Write-Host "Checking Python..."

$pythonOk = $false

if (Get-Command python -ErrorAction SilentlyContinue) {
    $version = python --version 2>&1
    if ($version -match "3\.11") {
        Write-Host "Python 3.11 already installed."
        $pythonOk = $true
    } else {
        Write-Host "Different Python version detected: $version"
    }
}

if (-not $pythonOk) {
    Write-Host "Installing Python 3.11..."

    $url = "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe"
    $output = "$env:TEMP\python-installer.exe"

    Invoke-WebRequest $url -OutFile $output

    Start-Process $output -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1" -Wait

    # Refresh PATH for current session
    $env:Path += ";C:\Program Files\Python311\Scripts;C:\Program Files\Python311"
}

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing uv..."
irm https://astral.sh/uv/install.ps1 | iex

# Ensure uv is available in current session
$env:Path += ";$HOME\.local\bin"

Write-Host "Verifying uv..."
uv --version

Write-Host "Creating virtual environment..."
uv venv

Write-Host "Activating virtual environment..."

try {
    . .venv\Scripts\Activate.ps1
} catch {
    Write-Host "Activation blocked. Run this once and re-run script:"
    Write-Host "Set-ExecutionPolicy RemoteSigned -Scope CurrentUser"
    exit 1
}

# Optional improvement: only install deps if project config exists
if (Test-Path "pyproject.toml") {
    Write-Host "Installing dependencies..."
    uv sync
} else {
    Write-Host "No pyproject.toml found. Skipping dependency installation."
}

Write-Host "Setup complete!"
Write-Host "Your environment is ready."