"""CPU identity checks and finite development exposure metadata; no native imports."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import h5py
import numpy as np

TASK = 'pick_dual_bottles'
SEEDS = (13000, 13001, 13002)
HDF5_KEY = 'trajectory_constraints/json'


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False)+'\n'
    with path.open('x', encoding='utf-8') as handle:
        handle.write(encoded)


def update_state(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def training_identity(root):
    """All provenance, plus task numeric trajectories/labels, not image-file hashes."""
    root = Path(root).resolve()
    rows, numeric, seeds, scenes, commits = [], [], set(), set(), set()
    files = sorted({*root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5'),
                    *root.glob('*/aloha-agilex_clean_50/raw/episode*.hdf5')})
    for path in files:
        stat = path.stat()
        with h5py.File(path, 'r') as handle:
            provenance = json.loads(handle.attrs['robotwin_expert_provenance'])
            if provenance['collection'] != 0 or provenance['task_config'] != 'demo_clean':
                raise ValueError('Non-clean training exclusion source')
            if path.parts[-4] == TASK:
                keys = ['endpose/'+arm+'_'+kind for arm in ('left', 'right') for kind in ('endpose', 'gripper')]
                arrays = {}
                for key in keys:
                    value = np.asarray(handle[key])
                    arrays[key] = dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
                embedded = handle[HDF5_KEY][()]
                if isinstance(embedded, bytes):
                    embedded = embedded.decode('utf-8')
                numeric.append(dict(path=str(path), arrays=arrays, bundle_sha256=digest(json.loads(str(embedded)))))
        seeds.add(int(provenance['seed'])); scenes.add(provenance['initial_scene_sha256']); commits.add(provenance['source_commit'])
        rows.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                         provenance_sha256=digest(provenance), seed=int(provenance['seed']),
                         scene_sha256=provenance['initial_scene_sha256']))
    if len(rows) != 2500 or len(numeric) != 50 or len(commits) != 1:
        raise ValueError('Expected exact2500clean provenance/50task arrays/single native revision')
    exclusions = dict(excluded_training_seeds=sorted(seeds), excluded_training_scene_sha256=sorted(scenes),
                      native_commit=commits.pop(), source_episodes=len(rows))
    return dict(version=1, root=str(root), exclusions=exclusions, provenance=rows, task_numeric=numeric,
                dataset_version_sha256=digest(dict(provenance=rows, task_numeric=numeric)),
                version_scope='provenance2500 plus50task numeric action/embedded-label hashes and file stats; not a claim to hash all image bytes')


def inspect_registry(registry_root):
    """Explicit root supplied by the plan; absence is scoped, never inferred authority."""
    root = Path(registry_root).resolve()
    path = root/'refinement_exposure'/f'{TASK}.json'
    result = dict(registry_root=str(root), task_ledger_path=str(path), exists=path.exists(),
                  sha256=None, ledger=None, excluded_seeds=[], excluded_scenes=[], receipt_sources=[])
    if not path.exists():
        return result
    value = read(path)
    if set(value) != {'version', 'task', 'reservations'} or value['version'] != 1 or value['task'] != TASK:
        raise ValueError('Malformed exposure registry')
    seeds, scenes = set(), set()
    identities = set()
    for entry in value['reservations']:
        if entry['state'] not in {'reserved', 'consumed'} or entry['reservation_id'] in identities:
            raise ValueError('Unknown/duplicate reservation state')
        identities.add(entry['reservation_id'])
        if not isinstance(entry['holdout_seeds'], list) or any(type(s) is not int for s in entry['holdout_seeds']):
            raise ValueError('Invalid reserved seeds')
        seeds.update(entry['holdout_seeds'])
        release_path = Path(entry['revision_root'])/'holdout_release.json'
        release = read(release_path)
        if digest(release) != entry['release_sha256']:
            raise ValueError('Registered release hash drift')
        result['receipt_sources'].append(dict(path=str(release_path), sha256=file_hash(release_path)))
        if entry['state'] == 'consumed':
            receipt_path = Path(entry['revision_root'])/'exposure_receipt.json'
            receipt = read(receipt_path)
            if file_hash(receipt_path) != entry['exposure_receipt_sha256']:
                raise ValueError('Consumed exposure receipt drift')
            scenes.update(receipt['scene_sha256'])
            result['receipt_sources'].append(dict(path=str(receipt_path), sha256=file_hash(receipt_path)))
    result.update(sha256=file_hash(path), ledger=value, excluded_seeds=sorted(seeds), excluded_scenes=sorted(scenes))
    return result


def exposure_scope(roots, *, ignore_roots=()):
    """Trace explicit registry fields and task ledgers, not guessed annotation roots."""
    ignore = [Path(p).resolve() for p in ignore_roots]
    filenames = {'experiment_manifest.json', 'manifest.json', 'search_manifest.json', 'configuration.json',
                 'plan.json', 'split_plan.json', 'design_plan.json', 'design.json', 'record.json',
                 'source_run.json', 'holdout_release.json', 'exposure_receipt.json', 'exposure_inventory.json',
                 'dataset_manifest.json'}
    seeds, scenes, sources, declared_registries, errors = set(), set(), [], [], []
    started = time.time()
    def numbers(value):
        if type(value) is int:
            return [value]
        if isinstance(value, list):
            return [n for v in value for n in numbers(v)]
        return []
    for root in map(Path, roots):
        if not root.is_dir():
            raise ValueError('Explicit exposure scan root is missing: '+str(root))
        for parent, dirs, files in os.walk(root):
            parent = Path(parent)
            dirs[:] = [d for d in dirs if d not in {'code', '.git', '__pycache__', 'raw', 'data', 'assets',
                                                   'native_scratch', 'node_modules', 'snapshots'}
                       and not any((parent/d).resolve().is_relative_to(p) for p in ignore)]
            if any(parent.resolve().is_relative_to(p) for p in ignore):
                continue
            for name in files:
                if name not in filenames and not (parent.name in {'refinement_exposure', 'development_registry'} and name.endswith('.json')):
                    continue
                path = parent/name
                try:
                    raw = path.read_bytes()
                    value = json.loads(raw)
                    source = dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
                    def visit(item, trail=''):
                        if isinstance(item, dict):
                            for key, val in item.items():
                                if 'seed' in key.lower():
                                    seeds.update(numbers(val))
                                if key in ('scene_sha256', 'initial_scene_sha256', 'excluded_training_scene_sha256'):
                                    vals = val if isinstance(val, list) else [val]
                                    scenes.update(v for v in vals if isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v))
                                if key == 'registry_root' and isinstance(val, str) and Path(val).is_absolute():
                                    declared_registries.append(dict(source, json_path=trail+'/'+key, root=str(Path(val).resolve())))
                                if 'seed_start' in key and type(val) is int:
                                    count = item.get('max_seed_attempts', item.get('seed_count'))
                                    if type(count) is int and 0 < count <= 100000:
                                        seeds.update(range(val, val+count))
                                visit(val, trail+'/'+key)
                        elif isinstance(item, list):
                            for i, val in enumerate(item):
                                visit(val, trail+'/'+str(i))
                    visit(value)
                    sources.append(source)
                except Exception as exc:
                    errors.append(dict(path=str(path), error=type(exc).__name__+': '+str(exc)))
    registries = [inspect_registry(p) for p in sorted({r['root'] for r in declared_registries})]
    for row in registries:
        seeds.update(row['excluded_seeds']); scenes.update(row['excluded_scenes'])
    if errors:
        raise ValueError('Incomplete exposure inventory: '+str(errors[:3]))
    return dict(version=1, scan_started_unix=started, scan_finished_unix=time.time(), roots=list(map(str, roots)),
                ignored_current_package_roots=list(map(str, ignore)), sources=sources,
                explicit_registry_fields=declared_registries, configured_registry_snapshots=registries,
                excluded_seeds=sorted(seeds), excluded_scene_sha256=sorted(scenes), read_errors=errors,
                scope='Explicit managed/data roots, declared registry_root fields and actual task-named exposure ledgers. Missing/unconfigured registry is scoped, never a global absence claim; new package development ledger is separately labelled.')


def assert_seed_exclusion(seeds, training, registry_rows, exposed_seeds):
    if tuple(seeds) != SEEDS:
        raise ValueError('Fixed3seeds cannot be replaced')
    forbidden = set(training['exclusions']['excluded_training_seeds']) | set(exposed_seeds)
    for row in registry_rows:
        forbidden.update(row['excluded_seeds'])
    if forbidden.intersection(seeds):
        raise ValueError('A fixed seed is training/reserved/consumed/exposed; no reseed')


def new_development_ledger(plan_sha256):
    return dict(version=1, kind='local_development_exposure_not_final_holdout_registry', task=TASK,
                plan_sha256=plan_sha256, state='all_fixed_seeds_reserved_no_retry',
                seeds=list(SEEDS), claims=[], independent_holdout=False)


def claim_scene(ledger_path, seed, scene_sha256, excluded_scenes):
    """Sequential workers append before expert motion, including failed captures."""
    path = Path(ledger_path)
    value = read(path)
    if value['kind'] != 'local_development_exposure_not_final_holdout_registry' or value['seeds'] != list(SEEDS):
        raise ValueError('Wrong development ledger')
    previous = value['claims']
    if seed not in SEEDS or any(r['seed'] == seed for r in previous):
        raise ValueError('Scene seed already claimed; no retry')
    if scene_sha256 in set(excluded_scenes) or any(r['scene_sha256'] == scene_sha256 for r in previous):
        raise ValueError('Fresh scene duplicates training/prior/current-campaign scene; no replacement')
    value['claims'].append(dict(seed=seed, scene_sha256=scene_sha256, state='exposed_before_expert'))
    update_state(path, value)
    return digest(value)
