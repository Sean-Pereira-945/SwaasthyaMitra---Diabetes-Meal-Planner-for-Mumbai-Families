$ErrorActionPreference = "Stop"

Write-Host "[Phase 1] Starting environment setup..."

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (!(Test-Path ".venv")) {
    Write-Host "Creating virtual environment at .venv"
    python -m venv .venv
}

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

Write-Host "Upgrading pip"
& $pythonExe -m pip install --upgrade pip

Write-Host "Installing requirements"
& $pythonExe -m pip install -r requirements.txt

Write-Host "Running phase 1 smoke test"
& $pythonExe scripts\phase1_smoke_test.py

Write-Host "[Phase 1] Setup complete."
Write-Host "To run app: .\.venv\Scripts\python.exe -m streamlit run app.py"
