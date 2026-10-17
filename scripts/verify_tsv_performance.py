"""Compare actual pre-change TSV adapter from the source snapshot, no OCR."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.recognizers.tesseract_recognizer import TesseractRecognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.baseline) as archive:
        name = next(n for n in archive.namelist() if n.replace('\\', '/').endswith('recognizers/tesseract_recognizer.py'))
        source = archive.read(name).decode('utf-8-sig')
    namespace = {'__name__': 'app.recognizers.baseline'}
    exec(compile(source, str(args.baseline), 'exec'), namespace)
    old, new = namespace['TesseractRecognizer'](), TesseractRecognizer()
    rows = []
    for count in (1000, 4000, 8000):
        data = dict(text=['Việt'] * count, conf=[90] * count, page_num=[1]*count,
                    block_num=[1]*count, par_num=[1]*count, line_num=[i//10 for i in range(count)])
        assert old._result(data, 'vie').to_dict() == new._result(data, 'vie').to_dict()
        measurements = {}
        for label, rec in [('before', old), ('after', new)]:
            runs = []
            for _ in range(5):
                start = time.perf_counter()
                rec._result(data, 'vie')
                runs.append(round((time.perf_counter()-start)*1000, 3))
            measurements[label] = dict(runs_ms=runs, median_ms=statistics.median(runs))
        rows.append(dict(words=count, equivalent=True, **measurements))
    print(json.dumps(dict(scope='TSV parsing only, same process/input, 5 repeats', results=rows), indent=2))


if __name__ == '__main__':
    main()
