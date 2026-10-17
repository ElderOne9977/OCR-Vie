# OCR-Vie

A local OCR application for Vietnamese and English. Choose an image in your browser, extract its text, and copy the result or download it as a plain text file.

The project combines PaddleOCR text detection, Tesseract recognition, and a FastAPI server. Image processing runs on your computer; the browser provides the upload and results interface. Internet access is needed to clone the repository and install dependencies. The two PaddleOCR models are included in Git.

## How it works

```text
Scene image → PaddleOCR detection → Text crops → Tesseract → Text
Document    → Page preprocessing → Tesseract              → Text
```

These are the default paths for Vietnamese text. Other languages and explicit engine selections can use a different path.

- **Automatic routing:** a heuristic chooses between scene and document processing. You can select the image type yourself if the automatic choice gives poor results.
- **Vietnamese scene text:** PaddleOCR locates text regions, and Tesseract reads the cropped lines. This is useful for signs, labels, and photographs containing text.
- **Document recognition:** Tesseract reads the whole page, with grayscale preprocessing by default.
- **Browser workspace:** preview the selected image, choose the language, inspect the result, and copy or export the text.
- **HTTP API:** send an image from another application and receive text, detected regions, and processing details.

The current interface uses Vietnamese labels. The bundled Latin recognizer does not cover the full Vietnamese character set, so keep the recognizer on **Automatic** for Vietnamese images.

## What you need

The launch scripts target **Windows and PowerShell**, using **Python 3.11**. The default configuration runs inference on the CPU and does not require a dedicated GPU.

You will also need:

- Git and Python 3.11, including the Windows `py` launcher.
- Tesseract OCR with the Vietnamese (`vie`) and English (`eng`) language data installed.
- Disk space for the Python environment and dependencies, in addition to the roughly 92 MiB of bundled model weights.
- A browser to use the interface. Node.js is optional; some interface regression tests use it.

The Python setup script installs the wrapper for Tesseract, but does not install the Tesseract executable or its language data.

## Setup

### 1. Clone the project

```powershell
git clone https://github.com/ElderOne9977/OCR-Vie.git
cd OCR-Vie
```

The model weights are stored in regular Git, so cloning also downloads them. There is no separate model download step for the default Vietnamese pipeline.

### 2. Install Tesseract and its languages

Install Tesseract OCR and include both the Vietnamese and English language packs. The application checks the usual Windows installation path:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

If your installation is elsewhere, set its location in PowerShell before running setup or starting the server:

```powershell
$env:TESSERACT_CMD = 'D:\Tesseract-OCR\tesseract.exe'
```

Replace the example path with the location of your own executable.

### 3. Install the Python dependencies

From the repository root:

```powershell
$env:MODELS_DIR = (Resolve-Path .\models).Path
$env:ALLOW_MODEL_DOWNLOAD = 'false'
.\scripts\setup.ps1
```

Setup creates `.venv311`, installs the application and test dependencies, and checks the environment. Run it again when the dependency files change. The server launcher does not install or update packages.

The bundled assets are:

| Asset | Location |
| --- | --- |
| PaddleOCR text detector | `models/official_models/PP-OCRv5_server_det/` |
| PaddleOCR Latin recognizer | `models/official_models/latin_PP-OCRv5_mobile_rec/` |

Paths are relative to the repository root. The default Vietnamese scene pipeline needs the detector and Tesseract; the Latin recognizer is included for explicit Paddle recognition. Tesseract language files are managed by your Tesseract installation, outside these model directories.

## Extract text

Open PowerShell in the project directory and start the server:

```powershell
$env:MODELS_DIR = (Resolve-Path .\models).Path
$env:ALLOW_MODEL_DOWNLOAD = 'false'
.\scripts\run_local.ps1
```

Set these environment variables again when using a new terminal. If you set `TESSERACT_CMD` during setup, set that variable again too. An existing `PADDLE_PDX_CACHE_HOME` takes precedence over `MODELS_DIR`; unset it if you want to use the bundled models instead of another cache.

Once the server is listening, open **http://127.0.0.1:8000/** in your browser. Select an image, choose the language, and press the recognition button. Start with the automatic image type; try the document or scene option if the result is poor.

The first request can take longer while the detector loads. When recognition finishes, review the text and use the copy or download controls. Check Vietnamese accents, amounts, dates, and identifiers before using the result elsewhere.

Press **Ctrl+C** in the server terminal to stop the application. To use another port, start it with `-Port 8002`. You can add `-Warmup` to load and check the configured scene pipeline before the server begins accepting requests.

## Use the API

With the server running, open **http://127.0.0.1:8000/docs** to explore the endpoints and try requests from your browser.

From PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/ocr?route=auto&lang=vie%2Beng" -F "file=@image.png"
```

Replace `image.png` with your image path. The response contains the recognized text, regions, warnings, and processing details. Use `/v1/ocr/text` for a plain text response. Set `route=document` or `route=scene` to choose a processing path explicitly.

`/health/live` checks whether the process is running. `/health/ready` reports readiness for the default scene pipeline and also includes document readiness. It can return HTTP 503 before the scene model has loaded; checking health does not load the model itself.

## Adjust the settings

Set environment variables in the same terminal before starting the server. Runtime defaults are defined in `app/core/config.py`.

| To change… | Set… |
| --- | --- |
| Model directory | `MODELS_DIR` |
| Whether missing models may be downloaded | `ALLOW_MODEL_DOWNLOAD` |
| Paddle CPU thread count | `CPU_THREADS` |
| Maximum simultaneous OCR requests per process | `MAX_CONCURRENT_OCR` |
| Maximum upload size in bytes | `MAX_UPLOAD_BYTES` |
| Document preprocessing | `DOCUMENT_PREPROCESS` (`gray`, `original`, or `binary`) |

Larger images, more text regions, and higher concurrency can increase memory use and processing time. Change one setting at a time and check the effect on both recognition quality and resource use.

## Troubleshooting

- **Python setup fails:** check that `py -3.11 --version` works. Use the project's `.venv311` environment rather than a different Python installation.
- **Tesseract is unavailable:** check the executable path and confirm that both `vie` and `eng` language data are installed. Set `TESSERACT_CMD` if needed.
- **Scene recognition reports a missing model:** check that `MODELS_DIR` points to the repository's `models` directory and that no older `PADDLE_PDX_CACHE_HOME` setting overrides it.
- **The Latin engine rejects Vietnamese:** select Automatic or Paddle detection with Tesseract. The bundled Latin model does not contain all Vietnamese characters.
- **The result is poor or empty:** try a sharper, well-lit image, crop to the relevant text, and select the image type yourself. Multi-column pages may need to be split into simpler sections.
- **The server is busy or a request takes too long:** wait before retrying, or use a smaller crop. A browser timeout does not necessarily stop an OCR operation already running on the server.
- **PowerShell blocks a script:** if you trust the checkout and your machine's policy permits it, launch that script in a separate process with `powershell -ExecutionPolicy Bypass -File .\scripts\run_local.ps1`.

## Project layout

```text
OCR-Vie/
├── app/                       # API, OCR pipeline, and browser interface
├── models/official_models/    # Bundled PaddleOCR inference artifacts
├── scripts/                   # Setup, launchers, checks, and performance tools
├── tests/                     # Unit and integration tests
├── benchmarks/                # Dataset and evaluation tools
├── benchmark_data/            # Synthetic fixtures
├── benchmark_data_quality_v2/ # Synthetic fixtures with revised annotations
├── requirements.txt           # Base dependencies
├── requirements-paddle.txt    # PaddleOCR dependencies
├── requirements-dev.txt       # Test dependencies
└── Dockerfile
```

Python environments, caches, private inputs, generated reports, and working notes are excluded from Git. A Dockerfile is included, but the container workflow has not yet been fully validated.

## Tests and evaluation

After setup, with `MODELS_DIR` set as above:

```powershell
.\scripts\test.ps1
```

The suite includes real-model integration tests. Node.js enables the JavaScript regression checks; those checks are skipped when Node is unavailable.

The two sample datasets contain 24 synthetic images in total. They help catch regressions but do not establish accuracy on real photographs or documents. For your own evaluation, use images you have permission to use, review their labels, and keep private inputs under `benchmark_incoming/` or `data/`. The benchmark commands support `--reviewed-only` to select reviewed samples.

Supported uploads are JPG, PNG, WebP, BMP, and TIFF, with a default limit of **10 MiB per image**. PDF input is not supported. Complex tables, multi-column reading order, and handwriting are not handled reliably by the current pipeline.

## Built with

- **PaddleOCR** for text detection and optional Latin recognition.
- **Tesseract** and **pytesseract** for Vietnamese and English recognition.
- **OpenCV**, **NumPy**, and **Pillow** for image processing.
- **FastAPI** and **Uvicorn** for the local server.
- **HTML, CSS, and JavaScript** for the browser interface.

The bundled PaddleOCR models retain their upstream model cards as `MODEL_CARD.txt`. Their Apache-2.0 license and attribution are included in [LICENSE-APACHE-2.0.txt](models/LICENSE-APACHE-2.0.txt) and [THIRD_PARTY_NOTICES.txt](models/THIRD_PARTY_NOTICES.txt). This license covers those third-party artifacts; OCR-Vie's own source code does not currently have a separate license.
