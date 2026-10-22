"""Installed source and immutable episode selection, without simulator imports."""
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

COLLECTIONS = {0: 'demo_clean', 1: 'demo_randomized'}


def source_root():
    value = os.environ.get('ROBOTWIN_PATH')
    if not value:
        raise ValueError('Set ROBOTWIN_PATH to the existing RoboTwin checkout on the simulator host')
    root = Path(value).expanduser().resolve()
    if not (root / 'envs' / '_base_task.py').is_file():
        raise ValueError(f'No RoboTwin envs/_base_task.py under {root}')
    return root


def task_names():
    try:
        root = source_root()
    except ValueError:
        return []
    return sorted(p.stem for p in (root / 'envs').glob('*.py') if not p.stem.startswith('_')
                  and p.stem not in {'robot', 'camera'})


def check_task(task):
    if not isinstance(task, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', task):
        raise ValueError('task must be a RoboTwin module name')


def step_limit(task):
    check_task(task)
    path = source_root() / 'task_config' / '_eval_step_limit.yml'
    values = yaml.safe_load(path.read_text())
    limit = values.get(task)
    if type(limit) is not int or limit < 1:
        raise ValueError(f'Missing positive native step limit for {task}')
    return limit


class EpisodeManifest:
    def __init__(self, path, task):
        self.path = Path(path).resolve()
        payload = self.path.read_bytes()
        self.sha256 = hashlib.sha256(payload).hexdigest()
        data = json.loads(payload)
        if data.get('schema_version') != 1 or data.get('task') != task:
            raise ValueError('RoboTwin episode manifest version/task mismatch')
        if data.get('selection') != 'native_expert_success':
            raise ValueError('Episode manifest must record native expert feasibility selection')
        self.source_commit = data.get('source_commit')
        if not isinstance(self.source_commit, str) or not re.fullmatch(r'[0-9a-f]{40}', self.source_commit):
            raise ValueError('Manifest requires a source commit for reproducible scene selection')
        self.collections = data['collections']
        for collection in ('0', '1'):
            entries = self.collections.get(collection)
            if not isinstance(entries, list) or not entries:
                raise ValueError('Manifest needs nonempty clean and randomized collections')
            seen = set()
            for entry in entries:
                if (type(entry.get('seed')) is not int or entry['seed'] < 0
                        or not isinstance(entry.get('instruction'), str) or not entry['instruction'].strip()
                        or entry.get('task_config') != COLLECTIONS[int(collection)]):
                    raise ValueError('Invalid seed, instruction or task_config in episode manifest')
                if entry['seed'] in seen:
                    raise ValueError('Duplicate seed within one collection')
                seen.add(entry['seed'])

    def entry(self, collection, index):
        if collection not in COLLECTIONS or type(index) is not int or index < 0:
            raise ValueError('Invalid collection/episode index')
        try:
            return dict(self.collections[str(collection)][index])
        except IndexError as exc:
            raise ValueError(f'Episode index {index} outside collection {collection}') from exc


def configured_manifest(task):
    path = os.environ.get('ROBOTWIN_EPISODE_MANIFEST')
    if not path:
        raise ValueError('Set ROBOTWIN_EPISODE_MANIFEST to a prepared, operator-owned manifest')
    return EpisodeManifest(path, task)
