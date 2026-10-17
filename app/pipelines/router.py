"""Hybrid OCR router.

Decides scene vs document, runs detector -> recognizer -> parser, giữ raw_text bất biến.
Không ghi đè raw bằng LLM. Không bịa confidence.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
from fastapi import HTTPException

from app.classifiers.base import InputClassifier
from app.classifiers.document_scene_classifier import HeuristicDocumentSceneClassifier
from app.core.config import get_settings
from app.pipelines.registry import get_detector, get_parser, get_recognizer
from app.preprocessing import document as doc_pre
from app.preprocessing import scene as scene_pre
from app.preprocessing.common import decode_bytes_to_bgr
from app.schemas.ocr import (
    BBox,
    ClassificationResult,
    PipelineResult,
    Provenance,
    RegionResult,
    TimingMs,
)
from app.services.errors import (
    CLASSIFICATION_FAILED,
    MODEL_UNAVAILABLE,
    RECOGNITION_FAILED,
    OCR_TIMEOUT,
    PipelineError,
)
from app.services.normalization import NORMALIZER_ID, normalize_text
from app.services.polygon_utils import crop_polygon, sort_reading_order
from app.services.limits import ocr_slot
from app.services.request_boundary import request_id
from app.services.paddle_models import is_vietnamese_lang

logger = logging.getLogger("vietnamese-ocr.router")


class OCRRouter:
    def __init__(self, classifier: InputClassifier | None = None) -> None:
        self.classifier = classifier or HeuristicDocumentSceneClassifier()

    # ---- public entry points -------------------------------------------------

    @ocr_slot()
    def process(
        self,
        image_bytes: bytes,
        *,
        route: str = "auto",
        recognizer: str = "auto",
        parser: str | None = None,
        lang: str = "vie+eng",
        psm: int = 6,
        filename: str | None = None,
        content_type: str | None = None,
        return_structure: bool = False,
    ) -> PipelineResult:
        started = time.perf_counter()
        image = decode_bytes_to_bgr(image_bytes)
        settings = get_settings()

        timing = TimingMs(decode=round((time.perf_counter() - started) * 1000, 2))

        # 1) Route.
        t0 = time.perf_counter()
        classification = self._resolve_route(
            route, image, filename=filename, content_type=content_type
        )
        timing.classification = round((time.perf_counter() - t0) * 1000, 2)

        warnings: list[str] = []
        if classification.route == "unknown":
            fallback = settings.fallback_route or "scene"
            warnings.append(
                f"classifier_low_confidence: fallback to '{fallback}' "
                f"(reason={classification.reason})"
            )
            classification = ClassificationResult(
                route=fallback,  # type: ignore[arg-type]
                confidence=classification.confidence,
                reason=classification.reason + "_fallback",
                classifier=classification.classifier,
            )

        if classification.route == "document":
            return self._run_document(
                image,
                classification,
                recognizer=recognizer,
                parser=parser,
                lang=lang,
                psm=psm,
                filename=filename,
                warnings=warnings,
                timing=timing,
                started=started,
                return_structure=return_structure,
            )
        return self._run_scene(
            image,
            classification,
            recognizer=recognizer,
            lang=lang,
            psm=psm,
            filename=filename,
            warnings=warnings,
            timing=timing,
            started=started,
        )

    # ---- route resolution ----------------------------------------------------

    def _resolve_route(
        self,
        route: str,
        image: np.ndarray,
        *,
        filename: str | None,
        content_type: str | None,
    ) -> ClassificationResult:
        key = (route or "auto").lower()
        if key in ("scene", "document"):
            return ClassificationResult(
                route=key,  # type: ignore[arg-type]
                confidence=1.0,
                reason="forced_by_request",
                classifier="manual",
            )
        if key not in ("auto", ""):
            raise PipelineError(
                CLASSIFICATION_FAILED,
                f"Unknown route '{route}'. Use auto|scene|document.",
                component="router",
            )
        try:
            return self.classifier.classify(
                image, filename=filename, content_type=content_type
            )
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("classification failed")
            raise PipelineError(
                "INTERNAL_ERROR",
                "Classification failed",
                component="router",
            ) from exc

    # ---- scene ---------------------------------------------------------------

    def _run_scene(
        self,
        image: np.ndarray,
        classification: ClassificationResult,
        *,
        recognizer: str,
        lang: str,
        psm: int,
        filename: str | None,
        warnings: list[str],
        timing: TimingMs,
        started: float,
    ) -> PipelineResult:
        settings = get_settings()
        recognizer = (recognizer or "auto").lower()
        engine_name = (recognizer if recognizer != "auto" else settings.scene_recognizer).lower()
        auto_selected = engine_name in ("auto", "")
        if engine_name in ("auto", "") and is_vietnamese_lang(lang):
            engine_name = "paddle_tesseract"
        rec_key = (engine_name or "").lower()
        explicit_tesseract = rec_key in ("tesseract", "tesseract_vie_eng")
        paddle_path = rec_key in ("paddle", "paddle_v5_latin", "paddle_v5", "pp-ocrv5", "latin_pp-ocrv5_mobile_rec", "paddle_tesseract")
        explicit_paddle = paddle_path and not auto_selected
        if rec_key not in ("auto", "", "tesseract", "tesseract_vie_eng") and not paddle_path:
            raise PipelineError(MODEL_UNAVAILABLE, f"Unsupported scene recognizer: {engine_name}", component="recognizer")
        # Scene uses Paddle detection; Vietnamese auto reads crops with Tesseract.
        # ``not_loaded`` is NOT a reason to skip Paddle (ensure_ready() loads it).
        # Explicit tesseract never tries neural; explicit paddle never silently falls back.
        want_neural = not explicit_tesseract

        fallback_used = False
        if want_neural:
            neural = self._run_scene_neural(
                image,
                classification,
                recognizer=engine_name,
                lang=lang,
                psm=psm,
                filename=filename,
                warnings=warnings,
                timing=timing,
                started=started,
            )
            if neural is not None:
                return neural
            # Neural path declined: dependency missing / weights absent / load failed.
            if explicit_paddle:
                raise PipelineError(
                    MODEL_UNAVAILABLE,
                    "Paddle scene OCR requested but detector/recognizer not ready.",
                    component="paddle",
                )
            if not settings.enable_tesseract_fallback:
                raise PipelineError(
                    MODEL_UNAVAILABLE,
                    "Paddle scene OCR unavailable and ENABLE_TESSERACT_FALLBACK=false. "
                    "Install paddleocr + PP-OCRv5 weights, or opt in to the legacy "
                    "Tesseract fallback via ENABLE_TESSERACT_FALLBACK=true.",
                    component="paddle",
                )
            warnings.append("PADDLE_UNAVAILABLE_USING_LEGACY_TESSERACT")
            fallback_used = True

        # Legacy / explicit Tesseract path only.
        t0 = time.perf_counter()
        rec = get_recognizer("tesseract")
        if rec.status() != "ready":
            raise PipelineError(
                MODEL_UNAVAILABLE,
                f"recognizer '{rec.name}' status={rec.status()}",
                component=rec.name,
            )

        timing.initialization = round((timing.initialization or 0) + (time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        processed = scene_pre.preprocess(image)
        timing.preprocessing = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        try:
            result = rec.recognize(processed, lang=lang, psm=psm)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("scene recognition failed")
            raise PipelineError(
                RECOGNITION_FAILED,
                f"Recognition failed: {exc}",
                component=rec.name,
            ) from exc
        timing.recognition = round((time.perf_counter() - t0) * 1000, 2)

        height, width = image.shape[:2]
        region = RegionResult(
            id=0,
            polygon=[[0.0, 0.0], [float(width), 0.0], [float(width), float(height)], [0.0, float(height)]],
            bbox=BBox(0, 0, int(width), int(height)),
            raw_text=result.raw_text,
            normalized_text=result.normalized_text,
            detection_confidence=None,
            ocr_confidence=result.ocr_confidence,
            recognizer=result.engine,
            raw_confidence=result.raw_confidence,
        )
        timing.total = round((time.perf_counter() - started) * 1000, 2)
        return PipelineResult(
            success=True,
            route=classification.to_dict(),
            engines={
                "classifier": classification.classifier,
                "detector": None,
                "recognizer": result.engine,
                "parser": None,
                "normalizer": NORMALIZER_ID,
            },
            text=result.normalized_text,
            regions=[region],
            document=None,
            warnings=warnings,
            provenance=Provenance(
                route="scene",
                parser=None,
                detector=None,
                recognizer=result.engine,
                normalizer=NORMALIZER_ID,
            ),
            timing_ms=timing,
            filename=filename,
            fallback_used=fallback_used,
        )

    # ---- document ------------------------------------------------------------

    def _run_document(
        self,
        image: np.ndarray,
        classification: ClassificationResult,
        *,
        recognizer: str,
        parser: str | None,
        lang: str,
        psm: int,
        filename: str | None,
        warnings: list[str],
        timing: TimingMs,
        started: float,
        return_structure: bool,
    ) -> PipelineResult:
        settings = get_settings()
        engine_name = recognizer if recognizer not in ("", "auto") else settings.document_recognizer
        engine_name = (engine_name or "tesseract").lower()
        if engine_name == "auto":
            engine_name = "tesseract"
        if engine_name not in ("tesseract", "tesseract_vie_eng"):
            # A page must be segmented before a line recognizer sees it.
            output = self._run_scene(
                image, classification, recognizer=engine_name, lang=lang, psm=psm,
                filename=filename, warnings=warnings, timing=timing, started=started,
            )
            output.provenance.route = "document"
            if return_structure:
                t0 = time.perf_counter()
                parsed = get_parser(parser).parse(
                    "\n".join(r.raw_text for r in output.regions if r.raw_text), filename=filename
                )
                output.document = parsed.to_dict()
                output.engines["parser"] = parsed.parser_engine
                output.provenance.parser = parsed.parser_engine
                timing.parsing = round((time.perf_counter() - t0) * 1000, 2)
            timing.total = round((time.perf_counter() - started) * 1000, 2)
            return output
        t0 = time.perf_counter()
        rec = get_recognizer("tesseract")
        if rec.status() != "ready":
            raise PipelineError(MODEL_UNAVAILABLE, "Document Tesseract recognizer unavailable", component=rec.name)

        detector = get_detector("full_frame")
        timing.initialization = round((time.perf_counter() - t0) * 1000, 2)
        t0 = time.perf_counter()
        processed = doc_pre.preprocess(image)
        timing.preprocessing = round((time.perf_counter() - t0) * 1000, 2)

        t0 = time.perf_counter()
        regions_raw = detector.detect(image)
        timing.detection = round((time.perf_counter() - t0) * 1000, 2)

        region_results: list[RegionResult] = []
        full_text_parts: list[str] = []
        t0 = time.perf_counter()
        for idx, det in enumerate(regions_raw):
            x, y, w, h = det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height
            crop = processed[y : y + h, x : x + w] if w > 0 and h > 0 else processed
            try:
                result = rec.recognize(crop, lang=lang, psm=psm)
            except PipelineError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("document recognition failed")
                raise PipelineError(
                    RECOGNITION_FAILED,
                    f"Recognition failed: {exc}",
                    component=rec.name,
                ) from exc
            full_text_parts.append(result.raw_text)
            region_results.append(
                RegionResult(
                    id=idx,
                    polygon=det.polygon,
                    bbox=det.bbox,
                    raw_text=result.raw_text,
                    normalized_text=result.normalized_text,
                    detection_confidence=det.detection_confidence,
                    ocr_confidence=result.ocr_confidence,
                    recognizer=result.engine,
                    raw_confidence=result.raw_confidence,
                )
            )
        timing.recognition = round((time.perf_counter() - t0) * 1000, 2)

        combined_raw = "\n".join(p for p in full_text_parts if p)
        combined_norm, _ = normalize_text(combined_raw)

        document_dict: dict[str, Any] | None = None
        parser_name: str | None = None
        if return_structure:
            t0 = time.perf_counter()
            doc_parser = get_parser(parser)
            try:
                parsed = doc_parser.parse(combined_raw, filename=filename)
            except PipelineError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("document parse failed")
                raise PipelineError(
                    "DOCUMENT_PARSE_FAILED",
                    f"Parse failed: {exc}",
                    component=doc_parser.name,
                ) from exc
            timing.parsing = round((time.perf_counter() - t0) * 1000, 2)
            parser_name = parsed.parser_engine
            document_dict = parsed.to_dict()

        timing.total = round((time.perf_counter() - started) * 1000, 2)
        return PipelineResult(
            success=True,
            route=classification.to_dict(),
            engines={
                "classifier": classification.classifier,
                "detector": detector.name,
                "recognizer": rec.name,
                "parser": parser_name,
                "normalizer": NORMALIZER_ID,
            },
            text=combined_norm,
            regions=region_results,
            document=document_dict,
            warnings=warnings,
            provenance=Provenance(
                route="document",
                parser=parser_name,
                detector=detector.name,
                recognizer=rec.name,
                normalizer=NORMALIZER_ID,
            ),
            timing_ms=timing,
            filename=filename,
            fallback_used=False,
        )

    # ---- neural scene (polygon + perspective rectify) ---------------------------
    def _run_scene_neural(
        self,
        image: np.ndarray,
        classification: ClassificationResult,
        *,
        recognizer: str,
        lang: str,
        psm: int,
        filename: str | None,
        warnings: list[str],
        timing: TimingMs,
        started: float,
    ) -> PipelineResult | None:
        """Neural (Paddle) scene path.

        Called for ``auto`` and explicit ``paddle*``. Returns a result when
        BOTH detector and recognizer become READY; otherwise returns ``None``
        so the caller decides: ``auto`` -> Tesseract (fallback_used=true),
        explicit paddle -> ``MODEL_UNAVAILABLE`` (no silent fallback).

        Status semantics (never equate not_loaded with unavailable):
          - NOT_LOADED / LOADING -> ``ensure_ready()`` attempts a real load.
          - READY -> use Paddle.
          - UNAVAILABLE -> dependency/weights absent; skip (caller may fallback).
          - LOAD_FAILED -> load raised; skip (caller may fallback).
        """
        t_init = time.perf_counter()
        detector = get_detector("paddle")
        hybrid = recognizer.lower() == "paddle_tesseract"
        rec = get_recognizer("tesseract" if hybrid else "paddle")

        det_state = detector.status()
        if det_state == "unavailable":
            warnings.append("detector 'paddle' unavailable; not usable")
            return None
        rec_state = rec.status()
        if rec_state == "unavailable":
            warnings.append(f"recognizer '{rec.name}' unavailable; not usable")
            return None
        if callable(getattr(detector, "ensure_ready", None)) and not detector.ensure_ready():
            warnings.append(f"detector 'paddle' status={detector.status()}; not usable")
            return None
        if callable(getattr(rec, "ensure_ready", None)) and not rec.ensure_ready(lang):
            warnings.append(f"recognizer '{rec.name}' status={rec.status()}; not usable")
            return None

        if rec.status() != "ready" or rec.name != ("tesseract" if hybrid else "paddle"):
            warnings.append(f"recognizer '{rec.name}' not usable in selected pipeline")
            return None

        # Product-facing engine label: adapter name is generic 'paddle', the
        # concrete PP-OCRv5 Latin engine is what provenance should report.
        rec_label = "paddle_v5_latin" if rec.name == "paddle" else rec.name
        timing.initialization = round((time.perf_counter() - t_init) * 1000, 2)

        t0 = time.perf_counter()
        try:
            detections = detector.detect(image)
        except PipelineError as exc:
            if exc.code == OCR_TIMEOUT:
                raise
            logger.exception('Paddle detection failed request_id=%s', request_id.get())
            warnings.append('paddle detection failed')
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception('Paddle detection failed request_id=%s', request_id.get())
            warnings.append('paddle detection error')
            return None
        timing.detection = round((time.perf_counter() - t0) * 1000, 2)

        if not detections:
            warnings.append("paddle detector returned 0 regions; returning empty result")
            timing.total = round((time.perf_counter() - started) * 1000, 2)
            return PipelineResult(
                success=True,
                route=classification.to_dict(),
                engines={
                    "classifier": classification.classifier,
                    "detector": detector.name,
                    "recognizer": rec_label,
                    "parser": None,
                    "normalizer": NORMALIZER_ID,
                },
                text="",
                regions=[],
                document=None,
                warnings=warnings,
                provenance=Provenance(
                    route="scene",
                    parser=None,
                    detector=detector.name,
                    recognizer=rec_label,
                    normalizer=NORMALIZER_ID,
                ),
                timing_ms=timing,
                filename=filename,
                fallback_used=False,
            )

        if len(detections) > get_settings().max_regions:
            raise HTTPException(413, detail="Detected region count exceeds MAX_REGIONS")
        detections = sort_reading_order(detections)
        region_results: list[RegionResult] = []
        full_text_parts: list[str] = []
        t0 = time.perf_counter()
        batch_results = {}
        batch_method = getattr(rec, "recognize_many", None)
        batch_mode = hybrid and callable(batch_method)
        pixels = 0
        pixel_budget = get_settings().max_image_pixels
        if batch_mode:
            crops, indices = [], []
            for idx, det in enumerate(detections):
                points = np.asarray(det.polygon)
                height, width = image.shape[:2]
                if not np.isfinite(points).all() or (points < 0).any() or (points[:, 0] > width).any() or (points[:, 1] > height).any():
                    raise PipelineError(RECOGNITION_FAILED, f"Invalid crop polygon for region {idx}", component=rec.name)
                crop = crop_polygon(image, det.polygon, max_pixels=pixel_budget - pixels)
                if crop is not None and crop.size:
                    pixels += crop.shape[0] * crop.shape[1]
                    crops.append(crop)
                    indices.append(idx)
            results = batch_method(crops, lang=lang, psm=7)
            if len(results) != len(indices):
                raise PipelineError(RECOGNITION_FAILED, "Recognizer batch result count mismatch", component=rec.name)
            batch_results = dict(zip(indices, results))
        for idx, det in enumerate(detections):
            result = batch_results.get(idx) if batch_mode else None
            try:
                if not batch_mode:
                    crop = crop_polygon(image, det.polygon, max_pixels=pixel_budget - pixels)
                    if crop is not None and crop.size:
                        pixels += crop.shape[0] * crop.shape[1]
                        result = rec.recognize(crop, lang=lang, psm=7 if hybrid else psm)
            except HTTPException:
                raise
            except PipelineError as exc:
                if hybrid or exc.code == OCR_TIMEOUT:
                    raise
                logger.exception('Region %s recognition failed request_id=%s', idx, request_id.get())
                warnings.append(f'region {idx} recognition failed')
            except Exception as exc:  # noqa: BLE001
                if hybrid:
                    raise PipelineError(RECOGNITION_FAILED, f"Recognition failed for region {idx}", component=rec.name) from exc
                logger.exception('Region %s recognition failed request_id=%s', idx, request_id.get())
                warnings.append(f'region {idx} recognition error')
            if result is not None:
                full_text_parts.append(result.raw_text)
            region_results.append(
                RegionResult(
                    id=idx,
                    polygon=det.polygon,
                    bbox=det.bbox,
                    raw_text=result.raw_text if result else "",
                    normalized_text=result.normalized_text if result else "",
                    detection_confidence=det.detection_confidence,
                    ocr_confidence=result.ocr_confidence if result else None,
                    recognizer=result.engine if result else rec_label,
                    raw_confidence=result.raw_confidence if result else None,
                )
            )
        timing.recognition = round((time.perf_counter() - t0) * 1000, 2)
        if not full_text_parts and any('recognition failed' in w or 'recognition error' in w for w in warnings):
            raise PipelineError(RECOGNITION_FAILED, "All regions failed recognition", component=rec.name)
        combined_raw = "\n".join(p for p in full_text_parts if p)
        combined_norm, _ = normalize_text(combined_raw)
        timing.total = round((time.perf_counter() - started) * 1000, 2)
        return PipelineResult(
            success=True,
            route=classification.to_dict(),
            engines={
                "classifier": classification.classifier,
                "detector": detector.name,
                "recognizer": rec_label,
                "parser": None,
                "normalizer": NORMALIZER_ID,
            },
            text=combined_norm,
            regions=region_results,
            document=None,
            warnings=warnings,
            provenance=Provenance(
                route="scene",
                parser=None,
                detector=detector.name,
                recognizer=rec_label,
                normalizer=NORMALIZER_ID,
            ),
            timing_ms=timing,
            filename=filename,
            fallback_used=False,
        )


_router: OCRRouter | None = None


def get_router() -> OCRRouter:
    global _router
    if _router is None:
        _router = OCRRouter()
    return _router

