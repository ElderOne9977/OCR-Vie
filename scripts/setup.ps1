$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$ocrPython = Join-Path $projectRoot '.venv311\Scripts\python.exe'
if (!(Test-Path -LiteralPath $ocrPython)) {
    & py -3.11 -m venv .venv311
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required.' }
}
# All OpenCV wheels share cv2 files; repair the old headless/contrib overlap.
& $ocrPython -m pip show opencv-python-headless opencv-python *> $null
if ($LASTEXITCODE -eq 0) {
    $ocrWheelCache = Join-Path ([System.IO.Path]::GetTempPath()) ('ocr-opencv-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $ocrWheelCache | Out-Null
    & $ocrPython -m pip download --no-deps --only-binary=:all: --dest $ocrWheelCache opencv-contrib-python==4.10.0.84
    if ($LASTEXITCODE -ne 0) { throw 'OpenCV wheel preparation failed; existing packages preserved.' }
    & $ocrPython -m pip uninstall -y opencv-python-headless opencv-python
    if ($LASTEXITCODE -ne 0) { throw 'Old OpenCV package removal failed.' }
    & $ocrPython -m pip install --force-reinstall --no-deps --no-index --find-links $ocrWheelCache opencv-contrib-python==4.10.0.84
    if ($LASTEXITCODE -ne 0) { throw 'OpenCV repair failed.' }
}
& $ocrPython -m pip install -r requirements-paddle.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $ocrPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
& $ocrPython scripts/check_environment.py
exit $LASTEXITCODE
