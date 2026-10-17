FROM python:3.11-slim

# Avoid prompts during apt operations.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Tesseract and language packs (vie + eng).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-vie \
        tesseract-ocr-eng \
        fonts-dejavu-core \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV MODELS_DIR=/models
ENV ALLOW_MODEL_DOWNLOAD=false
ENV OCR_WARMUP=true
RUN useradd --create-home --uid 10001 ocr && mkdir -p /models && chown ocr:ocr /models /app

# Copy dependency list first to leverage Docker cache.
COPY requirements.txt requirements-paddle.txt ./
RUN pip install --no-cache-dir -r requirements-paddle.txt

# Copy application source.
COPY app ./app
COPY scripts/smoke_test.py ./scripts/smoke_test.py

VOLUME ["/models"]
EXPOSE 8000
USER ocr

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/ready')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
