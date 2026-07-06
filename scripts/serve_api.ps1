# Sobe a API do Rosetta (FastAPI/uvicorn) com o modelo real, a partir da raiz do repo.
# Uso:  .\scripts\serve_api.ps1  [-Ckpt caminho.ckpt] [-Device cpu|cuda] [-Port 8000]

param(
    [string]$Ckpt = "checkpoints\crohme_rs\last.ckpt",
    [string]$Device = "cpu",
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo  # caminhos do config (vocab_path) são relativos à raiz

if (-not (Test-Path $Ckpt)) {
    Write-Warning "checkpoint '$Ckpt' não existe — a API sobe em modo stub (501)."
} else {
    $env:HMER_CKPT = $Ckpt
}
$env:HMER_DEVICE = $Device
$env:PYTHONPATH = "api\src;ml\src"

& "$repo\venv\Scripts\python.exe" -m uvicorn hmer_api.main:app --host 127.0.0.1 --port $Port

