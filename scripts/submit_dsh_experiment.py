#!/usr/bin/env python3
"""Submit one validated hosted DSH trial using a portable Slurm site profile."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from harness.deepseek_cli.auto_research import dsh_binary
from harness.deepseek_cli.login import Credential, MODEL
from harness.model_serving.lifecycle import write_json
from harness.model_serving.slurm import script
from scripts.run_deepseek_experiment import allocated_gpu
from services.dexjoco.tasks import OFFICIAL_TASKS


def submit(args):
    credential = Credential(args.provider, key_file=args.key_file, auth_home=args.auth_home)
    model = MODEL if args.provider == 'openrouter' else 'deepseek-flash'
    validation = json.loads(args.validation.read_text())
    if not (validation.get('status') == 'passed' and validation.get('provider') == args.provider
            and validation.get('model') == model and validation.get('wire_max_effort')
            and validation.get('vision_wire') and validation.get('resume')):
        raise ValueError('Pass matching native DSH provider/model validation before requesting an experiment GPU')
    binary = dsh_binary(args.binary)
    profile = json.loads(args.profile.read_text())
    resources = {'account': profile['account'], 'storage_group': profile.get('storage_group'),
                 **profile['experiment']}
    if any(word in resources['partition'].lower() for word in ('requeue', 'preempt')):
        raise ValueError('Use a regular GPU partition, not a requeue/preemptible partition')
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    manifest = root/'trial.json'
    python = PROJECT/'runtime/envs/host/bin/python'
    record = {'status': 'rendered', 'project': str(PROJECT), 'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT, text=True).strip(),
              'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=PROJECT, text=True).strip(),
              'created_at': time.time(), 'agent_cli': 'dsh', **credential.record(),
              'model': model, 'effort': 'max', 'task': args.task,
              'exploration_budget': 16, 'formal_episodes': 150, 'formal_workers': 4,
              'evaluation_mode': 'code_formal', 'exploration_envs': 1, 'subagents': 0,
              'observation_profile': 'rgbd', 'demonstration_context': 'terminal_state',
              'expert_demonstrations': 0, 'episode_reviews': False, 'external_perception': False,
              'validation': str(args.validation.resolve()), 'binary': str(binary),
              'resources': resources,
              'experiment_record': str(root/'experiment.json')}
    setup = ['cd '+shlex.quote(str(PROJECT)), 'export PYTHONDONTWRITEBYTECODE=1',
             'export PATH='+shlex.quote(str(Path(shutil.which('node')).parent))+':"$PATH"',
             'export RAX_RUNTIME_ROOT='+shlex.quote(str((PROJECT/'runtime').resolve())),
             'export RAX_ARTIFACT_ROOT="$RAX_RUNTIME_ROOT"',
             'export DEXJOCO_RUNTIME_ROOT="$RAX_RUNTIME_ROOT"', 'export TMPDIR=/tmp',
             'export OMP_NUM_THREADS=4', 'export OPENBLAS_NUM_THREADS=4',
             'export XDG_RUNTIME_DIR="/tmp/rax-dsh-${SLURM_JOB_ID}"',
             'mkdir -p "$XDG_RUNTIME_DIR"', 'chmod 700 "$XDG_RUNTIME_DIR"']
    path = root/'experiment.sbatch'
    path.write_text(script([python, Path(__file__).resolve(), 'run', '--manifest', manifest],
        name='rax-dsh-click-mouse', output=root/'slurm-%j.log', setup=setup, **resources))
    subprocess.run(['bash','-n',str(path)], check=True)
    write_json(manifest, record)
    if args.submit:
        record.update(status='submitted', job_id=subprocess.check_output(
            ['sbatch','--parsable',str(path)], text=True).strip().split(';')[0], submitted_at=time.time())
        write_json(manifest, record)
    print(json.dumps({'manifest': str(manifest), **record}, indent=2))


def experiment_command(record, gpu):
    """Keep account grants and API keys on their own authentication paths."""
    command = list(map(str, [PROJECT/'runtime/envs/host/bin/python', PROJECT/'scripts/run_deepseek_experiment.py',
        '--backend', 'dsh', '--provider', record.get('provider', 'openrouter'), '--model', record['model'],
        '--task', record['task'], '--agent-binary', record['binary'], '--sim-gpu', gpu,
        '--episodes', record['exploration_budget'], '--formal-episodes', record['formal_episodes'],
        '--record', record['experiment_record']]))
    if record.get('provider', 'openrouter') == 'deepseek-account':
        command += ['--agent-auth-home', record['auth_home']]
    else:
        command += ['--key-file', record['key_file']]
    return command


def run(args):
    record = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    if Path(record['experiment_record']).exists():
        raise FileExistsError('Trial already started; resume its session instead of duplicating it')
    gpu, python = allocated_gpu(), PROJECT/'runtime/envs/host/bin/python'
    with (root/'doctor.log').open('w') as log:
        subprocess.run([python, PROJECT/'scripts/dexjoco_doctor.py', '--gpu', gpu,
            '--task', record['task'], '--output', root/'doctor'], cwd=PROJECT,
            stdout=log, stderr=subprocess.STDOUT, check=True)
    command = experiment_command(record, gpu)
    os.execv(command[0], command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    launch = actions.add_parser('submit')
    for flag in ('root', 'profile', 'validation'):
        launch.add_argument('--'+flag, type=Path, required=True)
    launch.add_argument('--provider', choices=('openrouter', 'deepseek-account'), default='openrouter')
    launch.add_argument('--key-file', type=Path)
    launch.add_argument('--auth-home', type=Path)
    launch.add_argument('--binary', type=Path)
    launch.add_argument('--task', choices=sorted(OFFICIAL_TASKS), default='click_mouse')
    launch.add_argument('--submit', action='store_true')
    worker = actions.add_parser('run')
    worker.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    submit(args) if args.action == 'submit' else run(args)


if __name__ == '__main__':
    main()
