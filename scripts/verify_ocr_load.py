"""Loopback HTTP load/soak probe. Synthetic fixture, not a quality benchmark.

Run against an owned server: --server-pid PID --route scene --repeats 30
--concurrency 1 2 4 --soak-seconds 300. No model/data downloads.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.smoke_test import _get, _post_file, _simple_png

try:
    import psutil
except ImportError:
    psutil = None


def summarize(rows, seconds):
    timings = sorted(r['ms'] for r in rows if r['status'] == 200)
    busy = sum(r.get('error_code') == 'OCR_BUSY' for r in rows)
    return dict(requests=len(rows), successes=len(timings), busy=busy,
                errors=len(rows) - len(timings) - busy,
                timeouts=sum(r.get('error_code') == 'OCR_TIMEOUT' for r in rows),
                wrong_text=sum(r['status'] == 200 and not r['exact'] for r in rows),
                p50_ms=statistics.median(timings) if timings else None,
                p95_ms=timings[math.ceil(.95 * len(timings)) - 1] if timings else None,
                success_rps=len(timings) / seconds if seconds else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:8017')
    parser.add_argument('--server-pid', type=int, required=True)
    parser.add_argument('--route', choices=['document', 'scene'], default='scene')
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--soak-seconds', type=int, default=0)
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in ('127.0.0.1', 'localhost', '::1'):
        parser.error('Probe is restricted to loopback')
    if args.repeats < 1 or min(args.concurrency) < 1 or max(args.concurrency) > 16 or args.soak_seconds < 0:
        parser.error('Invalid repeat/concurrency/soak limits')
    png = _simple_png()
    status, health = _get(args.base_url, '/health/ready')
    try:
        server_config = json.loads(health)
    except ValueError:
        server_config = {'http_status': status, 'unavailable': True}
    proc = psutil.Process(args.server_pid) if psutil else None
    stop = threading.Event()
    samples, child_cpu = [], {}
    peak_temp = 0
    cpu_start = proc.cpu_times() if proc else None
    started = time.perf_counter()

    def sample():
        nonlocal peak_temp
        while not stop.is_set():
            rss = 0
            for p in [proc] + proc.children(recursive=True):
                try:
                    rss += p.memory_info().rss
                    if p.pid != proc.pid:
                        cpu = p.cpu_times()
                        child_cpu[(p.pid, p.create_time())] = cpu.user + cpu.system
                except psutil.Error:
                    pass
            samples.append([round(time.perf_counter() - started, 2), round(rss / 2**20, 2)])
            # Only our batch-prefix directories, never arbitrary user temp data.
            try:
                size = sum(p.stat().st_size for d in Path(tempfile.gettempdir()).glob('ocr_batch_*')
                           if d.is_dir() for p in d.iterdir() if p.is_file())
                peak_temp = max(peak_temp, size)
            except OSError:
                pass
            stop.wait(.1)

    def request(_=None):
        start = time.perf_counter()
        code, body = _post_file(args.base_url, f'/v1/ocr?route={args.route}&lang=vie%2Beng', png)
        error_code = None
        try:
            payload = json.loads(body)
            exact = ' '.join(payload.get('text', '').split()) == 'Xin chào Việt Nam'
            if isinstance(payload.get('detail'), dict):
                error_code = payload['detail'].get('code')
        except ValueError:
            exact = False
        return dict(status=code, ms=round((time.perf_counter() - start) * 1000, 2), exact=exact, error_code=error_code)

    sampler = threading.Thread(target=sample, daemon=True) if proc else None
    if sampler:
        sampler.start()
    groups = []
    try:
        first = request()
        print(json.dumps({'first_request': first}), file=sys.stderr, flush=True)
        for concurrency in args.concurrency:
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(pool.map(request, range(args.repeats)))
            group = dict(concurrency=concurrency, **summarize(rows, time.perf_counter() - start))
            groups.append(group)
            print(json.dumps(group), file=sys.stderr, flush=True)
        start = time.perf_counter()
        soak = []
        while time.perf_counter() - start < args.soak_seconds:
            soak.append(request())
        soak_summary = dict(seconds=time.perf_counter() - start, **summarize(soak, time.perf_counter() - start))
    finally:
        stop.set()
        if sampler:
            sampler.join()
    cpu_end = proc.cpu_times() if proc else None
    cpu = (cpu_end.user + cpu_end.system - cpu_start.user - cpu_start.system
           + sum(child_cpu.values())) if proc else None
    print(json.dumps(dict(fixture='synthetic_Arial_or_DejaVu_Xin_chao', image_sha256=hashlib.sha256(png).hexdigest(),
                         python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
                         server_config_before_first_request=server_config,
                         route=args.route, first_request=first, first_scope='cold only on a freshly started server',
                         warm=groups, soak=soak_summary, rss_tree_mib_samples=samples,
                         peak_tree_rss_mib=max((s[1] for s in samples), default=None),
                         cpu_seconds_est=cpu, cpu_scope='aggregate server tree; short-lived children may be missed',
                         peak_batch_temp_bytes_sampled=peak_temp if proc else None,
                         sampling_interval_seconds=.1, representative_quality=False), indent=2))
    return int(first['status'] != 200 or not first['exact'] or
               any(g['errors'] or g['wrong_text'] for g in groups) or soak_summary['errors'] or soak_summary['wrong_text'])


if __name__ == '__main__':
    raise SystemExit(main())
