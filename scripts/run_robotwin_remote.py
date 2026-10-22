#!/usr/bin/env python3
"""Run local Codex/RAX against an existing RoboTwin environment over SSH."""
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
import uuid

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def preflight():
    for name in ('codex', 'ssh', 'bwrap'):
        if not shutil.which(name):
            raise RuntimeError(f'{name} is required on the local operator machine')
    import msgpack  # noqa: F401
    import numpy  # noqa: F401
    import PIL  # noqa: F401
    import yaml  # noqa: F401
    import jinja2  # noqa: F401
    with tempfile.TemporaryDirectory(prefix='rax-preflight-') as tmp:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(str(Path(tmp)/'probe.sock'))
            except PermissionError as exc:
                raise RuntimeError('This shell forbids Unix sockets. Run this command in a normal local '
                                   'terminal; RAX cannot start inside this restricted command sandbox.') from exc
    subprocess.run(['bwrap', '--unshare-all', '--die-with-parent', '--ro-bind', '/', '/',
                    '--proc', '/proc', '--dev', '/dev', '/bin/true'], check=True)


def prepare(task, login_home):
    from services.robotwin.remote import configuration, command
    from services.robotwin.catalog import EpisodeManifest
    from harness.codex_cli.auto_research import chatgpt_login
    from harness.codex_cli.configuration import write_config
    auth = (login_home/'auth.json').resolve()
    chatgpt_login(auth)
    remote = configuration()
    # Only small operator metadata crosses here. No assets, weights or login tokens.
    query = ('import base64,json,sys,yaml; from pathlib import Path; '
             'print(json.dumps({"manifest":base64.b64encode(Path(sys.argv[1]).read_bytes()).decode(),'
             '"limits":yaml.safe_load(Path(sys.argv[2],"task_config/_eval_step_limit.yml").read_text())}))')
    metadata = json.loads(subprocess.check_output(command(remote, [remote['python'], '-c', query,
                           remote['manifest'], remote['robotwin_path']]), text=True, timeout=60))
    root = PROJECT/'runtime'/'remote-operator'/uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = root/f'{task}.json'
    manifest.write_bytes(base64.b64decode(metadata['manifest'], validate=True))
    EpisodeManifest(manifest, task)
    remote['task_step_limits'] = metadata['limits']
    resolved = root/'remote.json'
    resolved.write_text(json.dumps(remote, indent=2))
    resolved.chmod(0o600)
    os.environ['ROBOTWIN_REMOTE_CONFIG'] = str(resolved)
    os.environ['ROBOTWIN_EPISODE_MANIFEST'] = str(manifest)
    provider = root/'providers'
    provider.mkdir(mode=0o700, exist_ok=True)
    local = tomllib.loads((login_home/'config.toml').read_text()) if (login_home/'config.toml').exists() else {}
    if not local.get('model'):
        raise ValueError('Set model in the local Codex config before starting the experiment')
    write_config(provider/'local.config.toml', {'model':local['model'], 'robodojo':{'chatgpt_auth':str(auth)}})
    return provider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-config', type=Path, required=True)
    parser.add_argument('--task', default='adjust_bottle')
    parser.add_argument('--sim-gpu', default='0')
    parser.add_argument('--episodes', type=int, default=6)
    parser.add_argument('--formal-episodes', type=int, default=5)
    parser.add_argument('--agent-effort', default='xhigh')
    parser.add_argument('--login-home', type=Path, default=Path.home()/'.codex')
    parser.add_argument('--resume', type=Path, help='Existing operator-input.json to resume')
    parser.add_argument('--prepare-only', action='store_true', help='Check metadata and prepare configuration without launching')
    args = parser.parse_args()
    os.environ['ROBOTWIN_REMOTE_CONFIG'] = str(args.remote_config.resolve())
    os.environ['PYTHONPATH'] = str(PROJECT)
    if not args.prepare_only:
        preflight()
    provider = prepare(args.task, args.login_home.expanduser())
    if args.resume:
        config = args.resume.resolve()
    else:
        output = subprocess.check_output([sys.executable, str(PROJECT/'scripts/configure_auto_research.py'),
            '--benchmark', 'robotwin', '--task', args.task, '--sim-gpu', args.sim_gpu,
            '--episodes', str(args.episodes), '--formal-episodes', str(args.formal_episodes), '--formal-workers', '1',
            '--agent-provider', 'custom', '--external-perception', 'off', '--agent-effort', args.agent_effort],
            cwd=PROJECT, text=True)
        config = Path(output.strip().splitlines()[-1])
    print(f'Operator configuration: {config}', flush=True)
    argv = [sys.executable, '-m', 'harness.codex_cli.auto_research', '--config', str(config),
            '--provider-home', str(provider), '--profile', 'local', '--non-interactive']
    if args.resume:
        argv.append('--resume')
    if args.prepare_only:
        print('Prepared; run again without --prepare-only to start a fresh experiment.', flush=True)
        return
    # Foreground run: Ctrl+C invokes the harness cleanup; the remote owner watches
    # SSH stdin and cannot outlive a disconnected operator indefinitely.
    raise SystemExit(subprocess.call(argv, cwd=PROJECT))


if __name__ == '__main__':
    main()
