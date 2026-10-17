from app.parsers.fallback_parser import FallbackDocumentParser
from app.recognizers.tesseract_recognizer import TesseractRecognizer


def test_fallback_reading_order_references_real_blocks():
    for text in ('A\n\nB', '\nA\n', '', '   \n'):
        result = FallbackDocumentParser().parse(text)
        assert result.reading_order == [b.id for b in result.pages[0].blocks]


def test_tsv_grouping_does_not_rescan_text_length_per_word():
    class CountedWords(list):
        length_reads = 0

        def __len__(self):
            self.length_reads += 1
            return super().__len__()

    words = CountedWords(['A', '', 'B', 'C'])
    result = TesseractRecognizer()._result({
        'text': words, 'conf': ['90', '-1', 'nan', '80'],
        'line_num': [1, 1, 1, 2],
    }, 'vie+eng')
    assert result.raw_text == 'A B\nC'
    assert result.ocr_confidence == .85
    assert words.length_reads <= 2
