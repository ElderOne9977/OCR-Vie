"""Offline setup check: no Paddle import, model initialization, or downloads."""
import importlib.util
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    print(f'Python {platform.python_version()} ({sys.executable})')
    errors = []
    if sys.version_info[:2] != (3, 11):
        errors.append('Canonical hybrid runtime requires Python 3.11')
    for name in ('fastapi', 'uvicorn', 'pytesseract', 'PIL', 'cv2', 'numpy', 'python_multipart'):
        try:
            __import__(name)
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    try:
        from app.recognizers.tesseract_recognizer import TesseractRecognizer
        if TesseractRecognizer().status(lang='vie+eng') != 'ready':
            errors.append('Tesseract binary and vie/eng traineddata required')
        from app.core.config import get_settings
        from app.services.paddle_models import DET_MODEL, resolve_rec_model
        from app.services.paddle_runtime import is_model_present, _cache_home
        settings = get_settings()
        scene = settings.scene_recognizer
        if scene not in ('tesseract', 'tesseract_vie_eng'):
            for name in ('paddle', 'paddleocr', 'paddlex'):
                if importlib.util.find_spec(name) is None:
                    errors.append(f'{name}: missing')
            needed = [DET_MODEL]
            if scene not in ('auto', 'paddle_tesseract'):
                needed.append(resolve_rec_model('vie+eng'))
            for name in needed:
                if not name or not is_model_present(name):
                    errors.append(f'Model artifacts missing: {name}')
        print(f'Model cache: {_cache_home()} (no models loaded by this check)')
    except Exception as exc:
        errors.append(str(exc))
    for error in errors:
        print(f'ERROR: {error}', file=sys.stderr)
    print('Environment check: ' + ('FAILED' if errors else 'PASSED'))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
