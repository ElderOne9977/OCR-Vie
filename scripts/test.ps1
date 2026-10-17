param([switch]$Integration)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$ocrPython = Join-Path $projectRoot '.venv311\Scripts\python.exe'
if (!(Test-Path -LiteralPath $ocrPython)) { throw 'Run scripts/setup.ps1 first.' }
$env:ALLOW_MODEL_DOWNLOAD = 'false'
$testArgs = @('-m', 'pytest', '-q')
if ($Integration) { $testArgs += @('-m', 'integration') }
& $ocrPython @testArgs
exit $LASTEXITCODE
