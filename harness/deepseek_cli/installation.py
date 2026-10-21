"""Pinned native DSH installation, separate from providers and experiment state."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

VERSION = '0.2.1-rc.2'


def install(prefix, cache):
    prefix, cache = Path(prefix).resolve(), Path(cache).resolve()
    manifest = prefix/'node_modules/@deepseek-ai/dsh/package.json'
    if not manifest.exists():
        prefix.mkdir(parents=True, exist_ok=True)
        subprocess.run([shutil.which('npm') or 'npm', 'install', '--ignore-scripts', '--no-audit',
            '--no-fund', '--prefix', str(prefix), '--cache', str(cache),
            f'@deepseek-ai/dsh@{VERSION}'], check=True)
    if json.loads(manifest.read_text())['version'] != VERSION:
        raise ValueError(f'Use a separate installation prefix for DSH {VERSION}')
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Install Node.js 24 or newer before installing DSH')
    # This pinned helper only makes the packaged node-pty spawn helper executable.
    # Other dependency lifecycle scripts are unnecessary for the headless driver.
    helper = prefix/'node_modules/@deepseek-ai/dsh-subprocess-local/scripts/ensure-spawn-helper.mjs'
    subprocess.run([node, str(helper)], check=True)
    lock = prefix/'package-lock.json'
    record = {'version': VERSION, 'binary': str(prefix/'node_modules/.bin/dsh'), 'node': node,
              'lock_sha256': hashlib.sha256(lock.read_bytes()).hexdigest()}
    (prefix/'rax-installation.json').write_text(json.dumps(record, indent=2)+'\n')
    return record
