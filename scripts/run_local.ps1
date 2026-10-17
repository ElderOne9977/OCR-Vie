param([int]$Port = 8000, [switch]$Reload, [switch]$Warmup)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$ocrPython = Join-Path $projectRoot '.venv311\Scripts\python.exe'
if (!(Test-Path -LiteralPath $ocrPython)) { throw 'Run scripts/setup.ps1 first (Python 3.11 required).' }
if (!$env:ALLOW_MODEL_DOWNLOAD) { $env:ALLOW_MODEL_DOWNLOAD = 'false' }
if ($Warmup) { $env:OCR_WARMUP = 'true' }
$ocrArgs = @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$Port")
if ($Reload) { $ocrArgs += '--reload' }
& $ocrPython @ocrArgs
if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }
