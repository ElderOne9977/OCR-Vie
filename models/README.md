# Bundled PaddleOCR models

This repository includes unmodified inference artifacts for:

- `official_models/PP-OCRv5_server_det`: text detection, approximately 84 MiB of weights.
- `official_models/latin_PP-OCRv5_mobile_rec`: Latin text recognition, approximately 8 MiB of weights.

Model cards are included in each directory. See `THIRD_PARTY_NOTICES.md` and
`LICENSE-APACHE-2.0.txt` for attribution and license terms. The model cards are
upstream material, not validation of OCR-Vie's Vietnamese accuracy.

From the project root in PowerShell:

```powershell
$env:MODELS_DIR = (Resolve-Path .\models).Path
$env:ALLOW_MODEL_DOWNLOAD = 'false'
.\scripts\run_local.ps1
```

The Vietnamese scene pipeline uses the detector with the separately installed
Tesseract binary and `vie`/`eng` traineddata. The bundled Latin recognizer does
not contain the complete Vietnamese charset. Python dependencies, Tesseract and
traineddata are not bundled here.

Only these two official model directories are tracked. Local experiment models
and caches elsewhere under `models/` stay ignored. Files are stored in regular
Git, not Git LFS; each bundled file is below GitHub's 100 MiB limit.
