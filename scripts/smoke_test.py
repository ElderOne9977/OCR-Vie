#!/usr/bin/env python3
"""Minimal API smoke test using stdlib + urllib.

Usage: python scripts/smoke_test.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Add project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont


def _simple_png() -> bytes:
    font = None
    for path in (r'C:\Windows\Fonts\arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            font = ImageFont.truetype(path, 32)
            break
    if font is None:
        raise RuntimeError('Smoke needs Arial or DejaVuSans to render Vietnamese')
    im = Image.new("RGB", (450, 80), color="white")
    ImageDraw.Draw(im).text((12, 10), 'Xin chào Việt Nam', font=font, fill='black')
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _boundary_multipart(field: str, filename: str, data: bytes) -> tuple[str, bytes]:
    boundary = "----smoke-boundary-1234"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + data + tail
    return boundary, body


def _get(base: str, path: str) -> tuple[int, str]:
    url = base.rstrip("/") + path
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read(512 * 1024).decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:500]
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, str(exc)


def _post_file(base: str, path: str, png: bytes) -> tuple[int, str]:
    url = base.rstrip("/") + path
    boundary, body = _boundary_multipart("file", "smoke.png", png)
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return r.status, r.read(512 * 1024).decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:2000]
    except (urllib.error.URLError, TimeoutError) as exc:
        return 0, str(exc)


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument('--allow-unavailable', action='store_true', help='Diagnostic only; not a release gate')
    args = parser.parse_args()

    base = args.base_url
    failures = 0

    code, body = _get(base, "/")
    print(f"GET / -> {code}: UI {'ok' if 'OCR-Vie' in body else 'missing'}")
    if code != 200 or "OCR-Vie" not in body:
        failures += 1

    code, body = _get(base, "/api")
    print(f"GET /api → {code}: {body}")
    if code != 200:
        failures += 1

    code, body = _get(base, "/health/live")
    print(f"GET /health/live -> {code}: {body}")
    if code != 200:
        failures += 1

    png = _simple_png()
    for route in ('document', 'scene'):
        code, body = _post_file(base, f'/v1/ocr?route={route}&recognizer=auto&lang=vie%2Beng', png)
        print(f'POST /v1/ocr route={route} -> {code}: {body}')
        if code == 503 and args.allow_unavailable:
            print('DIAGNOSTIC: unavailable model accepted; this is NOT release validation')
            continue
        try:
            payload = json.loads(body)
            if code != 200 or not payload.get('success') or ' '.join(payload.get('text', '').split()) != 'Xin chào Việt Nam':
                failures += 1
        except ValueError:
            failures += 1
    code, body = _get(base, '/health/ready')
    print(f'GET /health/ready -> {code}: {body}')
    if not args.allow_unavailable:
        try:
            if code != 200 or not json.loads(body).get('ready'):
                failures += 1
        except ValueError:
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
