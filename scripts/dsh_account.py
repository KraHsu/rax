#!/usr/bin/env python3
"""Prepare the native DeepSeek account Web login; never export credential values."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness.deepseek_cli.auto_research import dsh_binary
from harness.model_serving.lifecycle import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('serve', 'status', 'stop'))
    parser.add_argument('--auth-home', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--port', type=int, default=3081)
    args = parser.parse_args()
    home = args.auth_home.expanduser().resolve()
    if args.action == 'status':
        path = home/'account-status.json'
        print(path.read_text() if path.exists() else json.dumps({'status': 'not_started'}))
        return
    if args.action == 'stop':
        record = json.loads((home/'login-service.json').read_text())
        pid = record['pid']
        command = Path(f'/proc/{pid}/cmdline')
        if command.exists():
            if str(home/'login-only.patch.yml').encode() not in command.read_bytes().split(b'\0'):
                raise RuntimeError('Recorded PID belongs to another process; refusing to stop it')
            os.kill(pid, signal.SIGTERM)
        record.update(stopped_at=time.time())
        write_json(home/'login-service.json', record)
        print(json.dumps({'status': 'stop_requested', 'pid': pid}))
        return
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    if home.stat().st_uid != os.getuid() or home.is_symlink():
        raise ValueError('Account directory must belong to the operator')
    home.chmod(0o700)
    record_path = home/'login-service.json'
    if record_path.exists():
        record = json.loads(record_path.read_text())
        try:
            os.kill(record['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            print(json.dumps(record, indent=2))
            return
    binary = dsh_binary(args.binary)
    patch = home/'login-only.patch.yml'
    write_json(patch, [
        {'id': 'approval', 'config': {'policy': 'never'}},
        {'id': 'permission', 'config': {'defaultPreset': 'rax',
            'presets': {'rax': {'sandbox': 'workspace-write', 'approval': 'never'}}}},
        *[{'id': name, 'disabled': True} for name in ('tool-web','tool-subagent','tool-subagent-fork',
           'tool-subagent-control','tool-workflow','tool-plugin-manager')],
        {'insert': [{'id': 'rax-account-status', 'name': str(PROJECT/'harness/deepseek_cli/account_status.mjs')}]}])
    command = ['nohup', shutil.which('node'), str(binary), '--profile', 'web', '--patch', str(patch),
               '--host', '127.0.0.1', '--port', str(args.port), '--no-open']
    env = {**os.environ, 'DSH_HOME': str(home), 'RAX_DSH_START_SIGNIN': '1',
           'RAX_DSH_CALLBACK_ORIGIN': f'http://localhost:{args.port}'}
    log = home/'login-service.log'
    with log.open('ab') as stream:
        child = subprocess.Popen(command, cwd=home, env=env, stdin=subprocess.DEVNULL,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    log.chmod(0o600)
    record = {'pid': child.pid, 'host': socket.getfqdn(), 'port': args.port,
              'url': f'http://localhost:{args.port}', 'auth_home': str(home),
              'binary': str(binary), 'created_at': time.time(), 'log': str(log),
              'ssh_forward': f'ssh -N -L {args.port}:127.0.0.1:{args.port} {os.environ.get("USER", "user")}@{socket.getfqdn()}'}
    write_json(record_path, record)
    record_path.chmod(0o600)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
