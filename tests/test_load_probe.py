from scripts import verify_ocr_load as probe


def test_load_summary_excludes_busy_from_success_latency():
    rows = [{'status': 200, 'ms': 10, 'exact': True},
            {'status': 200, 'ms': 30, 'exact': False},
            {'status': 503, 'ms': 1, 'exact': False, 'error_code': 'OCR_BUSY'},
            {'status': 503, 'ms': 1, 'exact': False, 'error_code': 'MODEL_UNAVAILABLE'}]
    result = probe.summarize(rows, 2)
    assert result['p50_ms'] == 20
    assert result['p95_ms'] == 30
    assert result['busy'] == 1
    assert result['success_rps'] == 1
    assert result['wrong_text'] == 1
    assert result['errors'] == 1
