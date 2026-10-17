"""Lang -> PP-OCRv5 model mapping (pure, no paddle needed)."""
from app.services.paddle_models import normalize_lang, resolve_all_models, resolve_rec_model


def test_vie_maps_to_latin():
    assert resolve_rec_model("vie+eng") == "latin_PP-OCRv5_mobile_rec"
    assert resolve_rec_model("vi") == "latin_PP-OCRv5_mobile_rec"
    assert resolve_rec_model("vie") == "latin_PP-OCRv5_mobile_rec"


def test_en_maps_to_en_model():
    assert resolve_rec_model("en") == "en_PP-OCRv5_mobile_rec"
    assert resolve_rec_model("eng") == "en_PP-OCRv5_mobile_rec"


def test_unknown_lang_returns_none():
    assert resolve_rec_model("xx_unknown") is None


def test_normalize_prefers_latin():
    assert normalize_lang("eng+vie") == "vi"
    assert normalize_lang("EN") == "en"


def test_det_model_shared():
    det, rec = resolve_all_models("vi")
    assert det == "PP-OCRv5_server_det"
    assert rec == "latin_PP-OCRv5_mobile_rec"