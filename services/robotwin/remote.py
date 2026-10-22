"""Operator-owned SSH configuration. Credentials and simulator source stay off the agent."""
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess


def configuration():
    path = os.environ.get('ROBOTWIN_REMOTE_CONFIG')
    if not path:
        return None
    data = json.loads(Path(path).read_text())
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', data['host']) or data['host'].startswith('-'):
        raise ValueError('Invalid SSH host')
    if type(data.get('port', 22)) is not int or not 1 <= data.get('port', 22) <= 65535:
        raise ValueError('Invalid SSH port')
    for key in ('rax_root', 'python', 'robotwin_path', 'manifest', 'dataset_root', 'output_root'):
        if not PurePosixPath(data[key]).is_absolute() or '\n' in data[key]:
            raise ValueError(f'Remote {key} must be an absolute path')
    timeout = data.get('startup_timeout_seconds', 300)
    if type(timeout) is not int or not 1 <= timeout <= 1800:
        raise ValueError('startup_timeout_seconds must be an integer in [1,1800]')
    return data


def ssh(config):
    return ['ssh', '-T', '-p', str(config.get('port', 22)), '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']


def command(config, argv):
    return [*ssh(config), config['host'], shlex.join(list(map(str, argv)))]


def gpu_inventory():
    config = configuration()
    args = ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader']
    return subprocess.check_output(command(config, args) if config else args, text=True, timeout=30)


def catalog():
    config = configuration()
    return config['task_step_limits'] if config else None


def launch_command(config, *, task, collection, index, run_id, sim_gpu, manifest_sha256):
    environment = {
        'CUDA_VISIBLE_DEVICES': str(sim_gpu), 'PYTHONUNBUFFERED': '1',
        'PYTHONPATH': ':'.join([config['rax_root'], *config.get('pythonpath', [])]),
        'ROBOTWIN_PATH': config['robotwin_path'], 'ROBOTWIN_DATASET_DIR': config['dataset_root'],
        **config.get('environment', {}),
    }
    output = str(PurePosixPath(config['output_root']) / run_id / 'sim')
    args = ['env', *(f'{key}={value}' for key, value in environment.items()), config['python'],
            '-m', 'services.robotwin.server', '--task', task, '--collection', str(collection),
            '--index', str(index), '--manifest', config['manifest'], '--manifest-sha256', manifest_sha256,
            '--output', output, '--watch-stdin']
    return command(config, args)
