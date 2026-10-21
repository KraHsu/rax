"""Render and submit an explicit prepare → serve → validate → experiment job graph."""
import json
import shlex
import subprocess
from pathlib import Path

from .lifecycle import write_json


def script(command, *, account, partition, name, cpus, memory, walltime, output, gpus=None, setup=(),
           storage_group=None):
    """No cluster paths in the command builder; resource policy lives in the site profile."""
    headers = ['#!/bin/bash', f'#SBATCH --account={account}', f'#SBATCH --partition={partition}',
               f'#SBATCH --job-name={name}', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
               f'#SBATCH --cpus-per-task={cpus}', f'#SBATCH --mem={memory}',
               f'#SBATCH --time={walltime}', f'#SBATCH --output={output}', '#SBATCH --signal=B:TERM@300',
               '#SBATCH --no-requeue']
    if gpus:
        headers.append(f'#SBATCH --gres=gpu:{gpus}')
    group_setup = []
    if storage_group:
        group = shlex.quote(storage_group)
        # Re-enter the entire batch script so setup, package caches and children
        # use the chosen group even when copytree preserves non-setgid modes.
        group_setup = [f'if [ "$(id -gn)" != {group} ]; then',
                       f'  exec sg {group} -c "exec bash $(printf \'%q\' "$0")"', 'fi']
    return '\n'.join([*headers, 'set -euo pipefail', *group_setup, *setup,
                      'exec '+shlex.join(map(str, command)), ''])


def submit_pipeline(*, project, root, config, profile, key_file, image, openrouter_key_file=None, submit=False):
    project, root, config = (Path(p).resolve() for p in (project, root, config))
    profile = json.loads(Path(profile).read_text())
    defaults = {'account': profile['account'], 'storage_group': profile.get('storage_group')}
    python = project/'runtime/envs/host/bin/python'
    prepare_python = project/'runtime/envs/model-serving/bin/python'
    common = ['cd '+shlex.quote(str(project)), 'export PYTHONDONTWRITEBYTECODE=1',
              'export TMPDIR=/tmp', 'export OMP_NUM_THREADS=4', 'export OPENBLAS_NUM_THREADS=4',
              'export RAX_RUNTIME_ROOT='+shlex.quote(str((project/'runtime').resolve())),
              'export RAX_ARTIFACT_ROOT="$RAX_RUNTIME_ROOT"',
              'export DEXJOCO_RUNTIME_ROOT="$RAX_RUNTIME_ROOT"',
              'export HF_HOME="$RAX_RUNTIME_ROOT/serving/cache/huggingface"',
              'export PIP_CACHE_DIR="$RAX_RUNTIME_ROOT/serving/cache/pip"',
              'export SINGULARITY_CACHEDIR="$RAX_RUNTIME_ROOT/serving/cache/singularity"',
              'export XDG_RUNTIME_DIR="/tmp/rax-ds-${SLURM_JOB_ID}"',
              'mkdir -p "$XDG_RUNTIME_DIR"', 'chmod 700 "$XDG_RUNTIME_DIR"']
    base = ['-m', 'harness.model_serving']
    options = ['--config', config, '--root', root]
    stages = {
        'prepare': ([prepare_python, *base, 'prepare', *options], profile['prepare'], [*common,
            f'test -x {shlex.quote(str(prepare_python))} || {shlex.quote(str(python))} -m venv {shlex.quote(str(prepare_python.parent.parent))}',
            f'{shlex.quote(str(prepare_python))} -m pip install huggingface_hub hf_xet']),
        'serve': ([python, *base, 'serve', *options, '--key-file', key_file], profile['serve'], common),
        'test': ([python, *base, 'test', *options, '--key-file', key_file, '--image', image], profile['test'], common),
        'experiment': ([python, project/'scripts/run_deepseek_experiment.py', '--provider', 'custom',
                        '--endpoint-file', root/'endpoint.json', '--server-key-file', key_file,
                        '--record', root/'experiment.json',
                        *(['--key-file', openrouter_key_file] if openrouter_key_file else [])], profile['experiment'], common),
    }
    stages['experiment'][0].extend(profile.get('experiment_args', ()))
    jobs = root/'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    record_path = jobs/'submitted.json'
    if submit and record_path.exists():
        raise FileExistsError(f'Pipeline already submitted; inspect {record_path}')
    ids = {}
    for stage, (command, resources, setup) in stages.items():
        path = jobs/f'{stage}.sbatch'
        path.write_text(script(command, name='rax-ds-'+stage, output=jobs/f'{stage}-%j.log',
                               setup=setup, **{**defaults, **resources}))
        subprocess.run(['bash', '-n', str(path)], check=True)
        if submit:
            dependency = {'serve': ('afterok', 'prepare'), 'test': ('after', 'serve'),
                          'experiment': ('afterok', 'test')}.get(stage)
            args = ['sbatch', '--parsable']
            if dependency:
                kind, parent = dependency
                args.append(f'--dependency={kind}:{ids[parent]}')
            ids[stage] = subprocess.check_output([*args, str(path)], text=True).strip().split(';')[0]
            write_json(record_path, ids)
    if submit:
        # A failed protocol test must not leave four expensive GPUs idle. Successful
        # experiments also release their dedicated service when they finish.
        for name, dependency in (('cleanup-test', f"afternotok:{ids['test']}"),
                                 ('cleanup-experiment', f"afterany:{ids['experiment']}")):
            resources = {**profile['test'], 'walltime': '00:10:00', 'cpus': 1, 'memory': '1G'}
            path = jobs/f'{name}.sbatch'
            path.write_text(script(['scancel', ids['serve']], name='rax-ds-'+name,
                output=jobs/f'{name}-%j.log', setup=common,
                **{**defaults, **resources}))
            ids[name] = subprocess.check_output(['sbatch', '--parsable', '--dependency='+dependency,
                                                 str(path)], text=True).strip().split(';')[0]
            write_json(record_path, ids)
    return {'jobs_directory': str(jobs), 'job_ids': ids}
