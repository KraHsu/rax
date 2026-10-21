#!/usr/bin/env python3
"""Run one simulation session with Claude Code or native DeepSeek Harness.

Use inside a GPU allocation. Credentials remain in private files; records contain paths only.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def allocated_gpu():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = dict(tuple(part.strip() for part in row.split(',')) for row in rows.splitlines() if row.strip())
    # On this cluster the device cgroup may expose only one GPU, renumbered to zero,
    # even though SLURM_JOB_GPUS still contains its physical node ordinal.
    if len(devices) == 1:
        return next(iter(devices.values()))
    selector = os.environ.get('SLURM_JOB_GPUS', os.environ.get('CUDA_VISIBLE_DEVICES', '')).split(',')[0]
    if selector in devices.values():
        return selector
    if selector in devices:
        return devices[selector]
    raise RuntimeError('Cannot determine the allocated GPU; pass --sim-gpu with its UUID')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('claude', 'dsh'), default='claude')
    parser.add_argument('--provider', choices=('openrouter', 'custom', 'deepseek-api', 'deepseek-account'), default='openrouter')
    parser.add_argument('--agent-binary', type=Path)
    parser.add_argument('--agent-auth-home', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--key-file', type=Path, help='OpenRouter key; required only for the hosted experiment')
    parser.add_argument('--endpoint-file', type=Path, help='Ready manifest written by the local serving pipeline')
    parser.add_argument('--server-key-file', type=Path)
    parser.add_argument('--sim-gpu', help='GPU UUID; defaults to the allocated visible device')
    parser.add_argument('--benchmark', default=None)
    parser.add_argument('--task', default=None)
    parser.add_argument('--image', help='Digest-pinned agent image on container branches')
    parser.add_argument('--episodes', type=int, default=16)
    parser.add_argument('--formal-episodes', type=int)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    if args.backend == 'claude' and args.provider not in ('openrouter', 'custom'):
        parser.error('The native DeepSeek authentication modes require --backend dsh')
    if args.provider in ('openrouter', 'deepseek-api') and not args.key_file:
        parser.error(f'{args.provider} requires --key-file')
    args.model = args.model or ('deepseek/deepseek-v4.1-flash-1017'
                               if args.provider == 'openrouter' else 'deepseek-flash')
    sys.path.insert(0, str(PROJECT))
    from services.benchmarks import DEFAULT, BENCHMARKS
    args.benchmark = args.benchmark or DEFAULT
    args.task = args.task or BENCHMARKS[args.benchmark].task_names()[0]
    args.sim_gpu = args.sim_gpu or allocated_gpu()
    endpoint = None
    if args.provider == 'custom':
        if not args.endpoint_file or not args.server_key_file:
            parser.error('custom requires --endpoint-file and --server-key-file')
        endpoint = json.loads(args.endpoint_file.read_text())
        if endpoint.get('status') != 'ready':
            raise RuntimeError('The model service has not passed its readiness check')
        args.model = endpoint['model']
    configure = [sys.executable, str(PROJECT/'scripts/configure_auto_research.py'),
                 '--benchmark', args.benchmark, '--task', args.task, '--sim-gpu', args.sim_gpu,
                 '--episodes', str(args.episodes), '--agent-cli', args.backend, '--agent-provider', args.provider,
                 '--agent-effort', 'max', '--exploration-envs', '1', '--subagents', '0',
                 '--external-perception', 'off',
                 '--observation-profile', 'rgbd',
                 '--expert-demonstrations', '0', '--episode-reviews', 'off']
    if args.formal_episodes is not None:
        configure += ['--formal-episodes', str(args.formal_episodes)]
    if args.image:
        configure += ['--image', args.image]
    output = subprocess.check_output(configure, cwd=PROJECT, text=True)
    config = Path(output.strip().splitlines()[-1])
    if not config.is_file():
        raise RuntimeError(f'Configuration was not created: {output}')
    configured = json.loads(config.read_text())
    record = {'benchmark': configured['benchmark'], 'config': str(config), 'session': str(config.parent), 'task': args.task, 'model': args.model,
              'provider': args.provider, 'agent_cli': args.backend, 'effort': 'max', 'episodes': args.episodes,
              'formal_episodes': configured['formal_episodes'], 'created_at': time.time(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    command = [sys.executable, '-m', 'harness.codex_cli.auto_research', '--config', str(config),
               '--non-interactive', '--agent-model' if args.backend == 'dsh' else '--claude-model', args.model]
    if args.backend == 'dsh':
        for flag, value in (('--agent-binary', args.agent_binary), ('--agent-auth-home', args.agent_auth_home)):
            if value:
                command += [flag, str(value)]
    if args.key_file:
        command += ['--key-file', str(args.key_file)]
    if endpoint:
        command += [('--agent-base-url' if args.backend == 'dsh' else '--claude-base-url'), endpoint['base_url'],
                    ('--agent-key-file' if args.backend == 'dsh' else '--claude-key-file'), str(args.server_key_file)]
    with (config.parent/'agent.log').open('a') as log:
        child = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, _: child.send_signal(number))
        result = child.wait()
    record.update(returncode=result, finished_at=time.time())
    args.record.write_text(json.dumps(record, indent=2))
    raise SystemExit(result)


if __name__ == '__main__':
    main()
