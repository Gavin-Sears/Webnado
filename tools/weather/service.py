"""Optional single-worker HRRR service. Put public deployments behind an authenticated, rate-limited proxy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from time import monotonic
from urllib.parse import urlparse

from prepare import identifier, prepare, validate


def serve(root, host, port, origin):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=1)
    jobs = {}
    limits = {}
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Access-Control-Allow-Origin', origin)
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()

        def do_POST(self):
            if self.path != '/jobs':
                return self.reply(404, {'error':'Unknown endpoint'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 < length < 4096:
                    raise ValueError('Invalid request size')
                settings = validate(json.loads(self.rfile.read(length)))
                key = identifier(settings)
                with lock:
                    if (root / key / 'manifest.json').exists():
                        return self.reply(200, {'id':key, 'status':'ready', 'manifest':f'data/{key}/manifest.json'})
                    if key in jobs and jobs[key]['status'] == 'failed':
                        del jobs[key]
                    if key in jobs:
                        return self.reply(202, dict(jobs[key]))
                    now = monotonic()
                    for address in list(limits):
                        limits[address] = [time for time in limits[address] if now-time < 3600]
                        if not limits[address]: del limits[address]
                    address = self.client_address[0]
                    if len(limits.get(address, [])) >= 6:
                        return self.reply(429, {'error':'Please retry later; preparation is limited to six new jobs per hour'})
                    if len(jobs) >= 128:
                        return self.reply(503, {'error':'Job history is full; the operator must restart the worker'})
                    if sum(job['status'] in ['queued','working'] for job in jobs.values()) >= 4:
                        return self.reply(429, {'error':'Weather worker is busy; please retry later'})
                    if len(list(root.glob('*/manifest.json'))) >= 20:
                        return self.reply(503, {'error':'Weather cache is full; the operator must remove unused cases'})
                    limits.setdefault(address, []).append(now)
                    jobs[key] = {'id':key, 'status':'queued', 'message':'Waiting for the weather worker'}
                    response = dict(jobs[key])
                def work():
                    def progress(message):
                        with lock: jobs[key].update(status='working', message=message)
                    try:
                        prepare(settings, root, progress)
                        with lock: jobs[key].update(status='ready', manifest=f'data/{key}/manifest.json')
                    except Exception as error:
                        with lock: jobs[key].update(status='failed', message=str(error))
                pool.submit(work)
                self.reply(202, response)
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.reply(400, {'error':str(error)})

        def do_GET(self):
            path = urlparse(self.path).path
            if path.startswith('/jobs/'):
                key = path.removeprefix('/jobs/')
                with lock: job = dict(jobs[key]) if key in jobs else None
                return self.reply(200 if job else 404, job or {'error':'Unknown job'})
            parts = path.strip('/').split('/')
            if len(parts) != 3 or parts[0] != 'data' or len(parts[1]) != 24 or any(c not in '0123456789abcdef' for c in parts[1]):
                return self.reply(404, {'error':'Unknown endpoint'})
            if parts[2] not in ['manifest.json','state.bin']:
                return self.reply(404, {'error':'Unknown file'})
            target = root / parts[1] / parts[2]
            if not target.is_file():
                return self.reply(404, {'error':'Weather is not ready'})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if target.suffix == '.json' else 'application/octet-stream')
            self.send_header('Content-Length', str(target.stat().st_size))
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Cache-Control', 'public, max-age=31536000, immutable')
            self.end_headers()
            with target.open('rb') as stream:
                while chunk := stream.read(1024*1024): self.wfile.write(chunk)

    try:
        ThreadingHTTPServer((host, port), Handler).serve_forever()
    finally:
        pool.shutdown(wait=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='tools/weather/data')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--origin', default='http://localhost:5173')
    args = parser.parse_args()
    serve(args.data, args.host, args.port, args.origin)
