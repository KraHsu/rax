#!/usr/bin/env python3
"""Render/submit independent API experiments, or run one Slurm array element."""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from harness.claude_cli.login import private_text
from harness.model_serving.lifecycle import write_json
from harness.model_serving.slurm import script
from services.dexjoco.demonstrations import cached_experts, keyframes
from services.dexjoco.tasks import OFFICIAL_TASKS
from scripts.run_deepseek_experiment import allocated_gpu


def prepare(args):
    tasks = args.task or sorted(set(OFFICIAL_TASKS) - {'click_mouse'})
    if len(set(tasks)) != len(tasks) or any(task not in OFFICIAL_TASKS for task in tasks):
        raise ValueError('Choose distinct official tasks without the _random suffix')
    if args.concurrency is not None and not 1 <= args.concurrency <= len(tasks):
        raise ValueError('An optional concurrency cap must be between one and the number of tasks')
    if not private_text(args.key_file).strip():
        raise ValueError('The credential file is empty')
    # Check installed visual references before requesting any GPU resources.
    for task in tasks:
        for demo in cached_experts(PROJECT/'runtime/reference-demos/dexjoco', task):
            keyframes(demo)
    profile = json.loads(args.profile.read_text())
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root/'batch.json'
    if manifest.exists():
        raise FileExistsError(f'Use a fresh batch directory; inspect {manifest}')
    record = {'created_at': time.time(), 'tasks': tasks, 'model': args.model,
              'provider': 'openrouter', 'agent_cli': 'claude', 'effort': 'max',
              'exploration_budget': 16, 'formal_episodes': 150, 'formal_workers': 4,
              'evaluation_mode': 'code_formal', 'exploration_envs': 1, 'subagents': 0,
              'observation_profile': 'rgbd', 'demonstration_context': 'terminal_state',
              'expert_demonstrations': 0, 'external_perception': False,
              'concurrency': args.concurrency, 'key_file': str(args.key_file.resolve()),
              'profile': profile, 'job_id': None}
    python = PROJECT/'runtime/envs/host/bin/python'
    setup = ['cd '+shlex.quote(str(PROJECT)), 'export PYTHONDONTWRITEBYTECODE=1',
             'export RAX_RUNTIME_ROOT='+shlex.quote(str((PROJECT/'runtime').resolve())),
             'export RAX_ARTIFACT_ROOT="$RAX_RUNTIME_ROOT"',
             'export DEXJOCO_RUNTIME_ROOT="$RAX_RUNTIME_ROOT"',
             'export TMPDIR=/tmp', 'export OMP_NUM_THREADS=4', 'export OPENBLAS_NUM_THREADS=4',
             'export XDG_RUNTIME_DIR="/tmp/rax-api-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"',
             'mkdir -p "$XDG_RUNTIME_DIR"', 'chmod 700 "$XDG_RUNTIME_DIR"']
    resources = {'account': profile['account'], 'storage_group': profile.get('storage_group'),
                 **profile['experiment']}
    batch_script = root/'experiment.sbatch'
    batch_script.write_text(script([python, Path(__file__).resolve(), 'run', '--batch', manifest],
        name='rax-ds-api', output=root/'slurm-%A_%a.log', setup=setup, **resources))
    subprocess.run(['bash', '-n', str(batch_script)], check=True)
    write_json(manifest, record)
    if args.submit:
        array = f'0-{len(tasks)-1}'
        if args.concurrency is not None:
            array += f'%{args.concurrency}'
        job_id = subprocess.check_output(['sbatch', '--parsable',
            '--array='+array, str(batch_script)], text=True).strip().split(';')[0]
        record.update(job_id=job_id, submitted_at=time.time(),
                      task_jobs={task: f'{job_id}_{index}' for index, task in enumerate(tasks)})
        write_json(manifest, record)
    print(json.dumps({'manifest': str(manifest), 'job_id': record['job_id'], 'tasks': tasks}, indent=2))


def run(args):
    batch = json.loads(args.batch.read_text())
    index = int(os.environ['SLURM_ARRAY_TASK_ID'])
    if not 0 <= index < len(batch['tasks']):
        raise ValueError('Slurm array index is outside the task list')
    task = batch['tasks'][index]
    output = args.batch.resolve().parent/task
    output.mkdir(parents=True, exist_ok=True)
    record = output/'experiment.json'
    if record.exists():
        raise FileExistsError(f'Experiment already started; inspect {record}')
    gpu = allocated_gpu()
    python = PROJECT/'runtime/envs/host/bin/python'
    with (output/'doctor.log').open('w') as log:
        subprocess.run([python, PROJECT/'scripts/dexjoco_doctor.py', '--gpu', gpu,
                        '--task', task, '--output', output/'doctor'], cwd=PROJECT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [str(python), str(PROJECT/'scripts/run_deepseek_experiment.py'),
               '--provider', 'openrouter', '--model', batch['model'], '--task', task,
               '--sim-gpu', gpu, '--key-file', batch['key_file'],
               '--episodes', str(batch['exploration_budget']),
               '--formal-episodes', str(batch['formal_episodes']), '--record', str(record)]
    os.execv(command[0], command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    submit = commands.add_parser('submit', help='Render; add --submit to start the array')
    submit.add_argument('--root', type=Path, required=True)
    submit.add_argument('--profile', type=Path, required=True)
    submit.add_argument('--key-file', type=Path, required=True)
    submit.add_argument('--model', default='deepseek/deepseek-v4.1-flash-1017')
    submit.add_argument('--task', action='append', help='Default: the ten tasks other than click_mouse')
    submit.add_argument('--concurrency', type=int, help='Optional array cap; default: only Slurm quota/resource limits')
    submit.add_argument('--submit', action='store_true')
    worker = commands.add_parser('run', help='Run the task selected by SLURM_ARRAY_TASK_ID')
    worker.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'submit':
        prepare(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
