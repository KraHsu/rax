#!/usr/bin/env python3
"""Serve the H200 dataset through a persistent local SSH tunnel, without copying data."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REPO = '/mnt/data/liangjiaqi/rax_train/rax'
DEFAULT_ROOT = '/mnt/data/liangjiaqi/rax_train/datasets/robotwin_clean50_v1'
DEFAULT_PYTHON = '/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python'
REMOTE_START = '''
import json, os, pathlib, subprocess, sys, time, urllib.error, urllib.request
repo, root, python, port = sys.argv[1:]
url = 'http://127.0.0.1:'+port+'/api/health'
def health():
    try:
        with urllib.request.urlopen(url, timeout=2) as response: return json.load(response)
    except (OSError, ValueError): return None
existing = health()
if existing:
    if existing.get('service') != 'robotwin-dataset-viewer' or existing.get('root') != root:
        raise SystemExit('Remote port belongs to another service/dataset')
else:
    runtime = pathlib.Path(repo)/'runtime/dataset_viewer'; runtime.mkdir(parents=True, exist_ok=True)
    with (runtime/'server.log').open('ab') as log:
        child = subprocess.Popen([python, '-m', 'services.dataset_viewer.server', '--root', root, '--port', port],
            cwd=repo, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            env={**os.environ, 'OPENBLAS_NUM_THREADS':'1', 'OMP_NUM_THREADS':'1'})
    (runtime/'server.pid').write_text(str(child.pid))
    for _ in range(40):
        if health(): break
        if child.poll() is not None: raise SystemExit('Dataset server exited; inspect '+str(runtime/'server.log'))
        time.sleep(.25)
    else: raise SystemExit('Dataset server did not become ready')
print(json.dumps(health()))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='root@vr.turbo-ai.com')
    parser.add_argument('--ssh-port', type=int, default=20379)
    parser.add_argument('--remote-repo', default=DEFAULT_REPO)
    parser.add_argument('--root', default=DEFAULT_ROOT)
    parser.add_argument('--remote-python', default=DEFAULT_PYTHON)
    parser.add_argument('--local-port', type=int, default=8766)
    parser.add_argument('--remote-port', type=int, default=9020)
    parser.add_argument('--tunnel-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    ssh = ['ssh', '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-o', 'BatchMode=yes',
           '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
           '-p', str(args.ssh_port)]
    if args.tunnel_worker:
        # Dedicated connection: does not disturb collection or annotation SSH masters.
        while True:
            subprocess.run([*ssh, '-N', '-o', 'ExitOnForwardFailure=yes', '-L',
                            f'127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}', args.host])
            time.sleep(3)
    runtime = Path.home()/'.cache/rax-dataset-viewer'; runtime.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-az', '--exclude=__pycache__', '-e', shlex.join(ssh),
                    str(REPO/'services/dataset_viewer')+'/',
                    f'{args.host}:{shlex.quote(args.remote_repo)}/services/dataset_viewer/'], check=True, timeout=90)
    command = shlex.join(['python3', '-', args.remote_repo, args.root, args.remote_python, str(args.remote_port)])
    subprocess.run([*ssh, args.host, command], input=REMOTE_START, text=True, check=True, timeout=45)
    url = f'http://127.0.0.1:{args.local_port}'
    def local_health():
        try:
            with urlopen(url+'/api/health', timeout=2) as response:
                data = json.load(response)
            if data.get('service') != 'robotwin-dataset-viewer' or data.get('root') != args.root:
                raise SystemExit('Local port belongs to another service/dataset; choose --local-port')
            return True
        except (OSError, ValueError): return False
    if not local_health():
        # Fail before spawning a reconnect loop if an unrelated service owns the port.
        with socket.socket() as probe:
            try: probe.bind(('127.0.0.1', args.local_port))
            except OSError: raise SystemExit('Local port is occupied; choose --local-port')
        worker_args = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--tunnel-worker']
        with (runtime/'tunnel.log').open('ab') as log:
            worker = subprocess.Popen(worker_args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        (runtime/'tunnel.pid').write_text(str(worker.pid))
        for _ in range(40):
            if local_health(): break
            time.sleep(.5)
        else: raise SystemExit('Tunnel is reconnecting; inspect '+str(runtime/'tunnel.log'))
    print('Open '+url, flush=True)


if __name__ == '__main__': main()
