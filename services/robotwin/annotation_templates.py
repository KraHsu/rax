"""Task template registry; activate only after three distinct clean references."""
import json
from pathlib import Path
import time

import h5py

from services.robotwin.collection import atomic_json
from services.robotwin.programmatic_annotation import OutsideTemplate, instantiate, template_hash, validate_template


def entries(root, task):
    return [(p, json.loads(p.read_text())) for p in sorted((Path(root)/'annotation_templates'/task).glob('*.json'))]


def install(root, template, *, source_episode, agent_run):
    validate_template(template)
    source_episode = Path(source_episode).resolve()
    # The template must also pass on the very episode reviewed by RAX.
    with h5py.File(source_episode, 'r') as h:
        instantiate(h, template)
    path = Path(root)/'annotation_templates'/template['task']/(template_hash(template)+'.json')
    if not path.exists():
        atomic_json(path, {'template': template, 'status': 'candidate', 'source_episode': str(source_episode),
                          'agent_run': str(agent_run), 'created_unix': time.time(), 'validation': []})
    refresh(root, template['task'])
    return json.loads(path.read_text())


def refresh(root, task):
    directory = Path(root)/task/'aloha-agilex_clean_50'
    paths = sorted([*(directory/'raw').glob('episode*.hdf5'), *(directory/'data').glob('episode*.hdf5')],
                   key=lambda p: int(p.stem.removeprefix('episode')))
    for registry, entry in entries(root, task):
        if entry['status'] != 'candidate': continue
        audit = []; seen = set(); seeds = set()
        for path in paths:
            try:
                with h5py.File(path, 'r') as h:
                    constraints, result = instantiate(h, entry['template'])
                    meta = json.loads(h.attrs['robotwin_expert_provenance'])
                    fingerprint = meta['initial_scene_sha256']
                    if fingerprint in seen or meta['seed'] in seeds: continue
                    seen.add(fingerprint); seeds.add(meta['seed'])
                    audit.append({'episode': str(path), 'seed': meta['seed'], 'initial_scene_sha256': fingerprint,
                                  'constraints': len(constraints), 'signature': result['signature']})
            except (OutsideTemplate, FileNotFoundError, BlockingIOError):
                continue
            except (ValueError, KeyError) as exc:
                entry.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
                break
            if len(audit) >= 3:
                entry.update(status='active', activated_unix=time.time())
                break
        entry['validation'] = audit
        atomic_json(registry, entry)


def match(root, task, episode, *, include_candidates=False):
    statuses = ('active', 'candidate') if include_candidates else ('active',)
    candidates = [e for _, e in entries(root, task) if e['status'] in statuses]
    if not candidates: return None
    with h5py.File(episode, 'r') as h:
        for entry in candidates:
            try:
                constraints, audit = instantiate(h, entry['template'])
                return entry, constraints, audit
            except OutsideTemplate:
                continue
    return None
