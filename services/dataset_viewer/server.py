"""Loopback-only, GET-only dataset browser. No collection controls or writes."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

from services.dataset_viewer.store import Dataset

STATIC = Path(__file__).parent/'static'
ASSETS = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'application/javascript; charset=utf-8'),
          '/style.css': ('style.css', 'text/css; charset=utf-8')}


class Handler(BaseHTTPRequestHandler):
    dataset: Dataset
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_): pass

    def _headers(self, status, kind, length, extra=()):
        self.send_response(status)
        for k, v in [('Content-Type', kind), ('Content-Length', str(length)), ('Cache-Control', 'no-store'),
                     ('X-Content-Type-Options', 'nosniff'), ('X-Frame-Options', 'DENY'),
                     ('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; media-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"), *extra]:
            self.send_header(k, v)
        self.end_headers()

    def send(self, status, data, kind='application/json; charset=utf-8', extra=()):
        self._headers(status, kind, len(data), extra)
        try: self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError): pass

    def json(self, value, status=200):
        self.send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode())

    def do_GET(self):
        host = urlsplit('http://'+self.headers.get('Host', '')).hostname
        if host not in ('127.0.0.1', 'localhost', '::1') or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.json({'error': 'Loopback access only'}, 403); return
        request = urlsplit(self.path); query = {k: v[0] for k, v in parse_qs(request.query).items()}
        try:
            if request.path in ASSETS:
                name, kind = ASSETS[request.path]; self.send(200, (STATIC/name).read_bytes(), kind)
            elif request.path == '/api/health':
                self.json({'service': 'robotwin-dataset-viewer', 'dataset': self.dataset.root.name, 'root': str(self.dataset.root)})
            elif request.path == '/api/summary': self.json(self.dataset.summary())
            elif request.path == '/api/episodes': self.json(self.dataset.episodes(query.get('task', '')))
            elif request.path == '/api/episode': self.json(self.dataset.episode(query.get('task', ''), int(query.get('episode', '-1'))))
            elif request.path == '/api/frame':
                self.send(200, self.dataset.frame(query.get('task', ''), int(query.get('episode', '-1')),
                          int(query.get('frame', '-1')), query.get('camera', 'head_camera'), int(query.get('width', '640'))), 'image/jpeg')
            elif request.path == '/api/video': self.video(self.dataset.video_path(query.get('task', ''), int(query.get('episode', '-1'))))
            else: self.json({'error': 'Not found'}, 404)
        except FileNotFoundError as exc: self.json({'error': str(exc)}, 404)
        except (ValueError, KeyError) as exc: self.json({'error': str(exc)}, 400)
        except OSError: self.json({'error': 'Artifact is being published; please retry.'}, 503)

    def do_POST(self): self.json({'error': 'This dataset viewer is read-only'}, 405)

    def video(self, path):
        size = path.stat().st_size; start, end, status = 0, size-1, 200
        header = self.headers.get('Range')
        if header:
            m = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
            if m and any(m.groups()):
                left, right = m.groups()
                if left: start, end = int(left), min(int(right), size-1) if right else size-1
                else: start = max(0, size-int(right))
                status = 206
            if not m or not any(m.groups()) or start > end or start >= size:
                self.send(416, b'', 'video/mp4', [('Content-Range', f'bytes */{size}')]); return
        extra = [('Accept-Ranges', 'bytes')]
        if status == 206: extra.append(('Content-Range', f'bytes {start}-{end}/{size}'))
        self._headers(status, 'video/mp4', end-start+1, extra)
        try:
            with path.open('rb') as stream:
                stream.seek(start); remaining = end-start+1
                while remaining:
                    chunk = stream.read(min(1024*1024, remaining))
                    if not chunk: break
                    self.wfile.write(chunk); remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True); parser.add_argument('--port', type=int, default=9020)
    args = parser.parse_args(); Handler.dataset = Dataset(args.root)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler); server.daemon_threads = True
    print(f'RoboTwin Dataset Atlas: http://127.0.0.1:{args.port}', flush=True)
    try: server.serve_forever()
    finally: server.server_close()


if __name__ == '__main__': main()
