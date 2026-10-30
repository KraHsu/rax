#!/usr/bin/env python3
"""Publish a new clean dataset only from counterfactually accepted templates.

No simulator or model is started. Source HDF5s are read-only; every destination
gets an independent inode. A hidden staging directory is resumable, but the
requested output appears only after all 50 x 50 files and proofs validate.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py

from services.robotwin.acceptance import (
    ACCEPTANCE_VERSION, annotation_policy_fingerprint, assess_evidence, derived_values_equal, evaluator_fingerprint,
    file_hash, json_hash,
)
from services.robotwin.annotation import embed_review
from services.robotwin.annotation_templates import match, registry_readiness
from services.robotwin.collection import atomic_json
from services.robotwin.constraint_schema import read_embedded
from services.robotwin.constraint_refinement import refinement_readiness, validate_frozen_gate
from services.robotwin.programmatic_annotation import template_hash

EXPECTED_TASKS = 50
EPISODES_PER_TASK = 50
PUBLISHER_VERSION = 1
PROOF_KEY = 'trajectory_constraints/acceptance'


def read_json(path):
    return json.loads(Path(path).read_text())


def contained_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Publication paths must remain within the dataset')
    result = Path(root)/path
    if Path(root).resolve() not in result.resolve().parents:
        raise ValueError('Publication path escapes the dataset')
    return result


def source_inventory(root):
    root = Path(root).resolve()
    manifest = read_json(root/'batch_manifest.json')
    tasks = manifest['tasks']
    if (manifest.get('collection') != 'clean' or manifest.get('per_task') != EPISODES_PER_TASK
            or len(tasks) != EXPECTED_TASKS or len(set(tasks)) != EXPECTED_TASKS
            or any(not isinstance(task, str) or Path(task).name != task for task in tasks)):
        raise ValueError('Source must contain exactly 50 clean tasks x 50 episodes')
    if manifest.get('running') or manifest.get('pending_tasks') or manifest.get('failed_tasks'):
        raise ValueError('Source collection is not complete')
    inventory = []
    expected = {f'episode{i}.hdf5' for i in range(EPISODES_PER_TASK)}
    for task in tasks:
        paths = sorted((root/task/'aloha-agilex_clean_50/data').glob('episode*.hdf5'))
        if {p.name for p in paths} != expected:
            raise ValueError(f'Incomplete or unexpected episode quota: {task}')
        seeds, scenes = set(), set()
        for path in paths:
            with h5py.File(path, 'r') as handle:
                meta = json.loads(handle.attrs['robotwin_expert_provenance'])
                bundle = read_embedded(handle)
                if (meta.get('task') != task or meta.get('collection') != 0
                        or meta.get('task_config') != 'demo_clean' or meta.get('native_success') is not True
                        or not bundle or bundle['provenance'].get('native_success') is not True):
                    raise ValueError(f'Not a successful clean training recording: {path}')
                seed, scene = meta['seed'], meta['initial_scene_sha256']
                if type(seed) is not int or not isinstance(scene, str) or not scene:
                    raise ValueError(f'Invalid source scene identity: {path}')
                if seed in seeds or scene in scenes:
                    raise ValueError(f'Duplicate initial state in task {task}')
                seeds.add(seed); scenes.add(scene)
            inventory.append(dict(path=str(path.relative_to(root)), task=task, seed=seed,
                                  initial_scene_sha256=scene, source_sha256=file_hash(path),
                                  source_bytes=path.stat().st_size))
    all_paths = {str(p.relative_to(root)) for p in root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5')}
    if all_paths != {row['path'] for row in inventory}:
        raise ValueError('Unexpected source task or episode')
    return manifest, inventory


def select_plan(source_root, registry_root, *, require_refinement=False):
    manifest, inventory = source_inventory(source_root)
    reports, required, entries = {}, {}, []
    identities = {}
    for row in inventory:
        identities.setdefault(row['task'], dict(seeds=set(), scenes=set()))
        identities[row['task']]['seeds'].add(row['seed'])
        identities[row['task']]['scenes'].add(row['initial_scene_sha256'])
    for row in inventory:
        matched = match(registry_root, row['task'], Path(source_root)/row['path'])
        if matched is None:
            raise ValueError(f'No accepted matching template for {row["path"]}')
        entry, _, audit = matched
        digest = template_hash(entry['template'])
        if audit['template_sha256'] != digest:
            raise ValueError('Instantiation template identity changed')
        required.setdefault(row['task'], set()).add(digest)
        proof = entry['acceptance']
        report_key = f'{row["task"]}/{digest}'
        if report_key not in reports:
            report = read_json(proof['report_path'])
            evidence = read_json(Path(proof['evidence_root'])/'experiment_manifest.json')
            identity = identities[row['task']]
            if (not identity['seeds'] <= set(evidence['excluded_training_seeds'])
                    or not identity['scenes'] <= set(evidence['excluded_training_scene_sha256'])):
                raise ValueError('Acceptance evidence does not exclude every training scene/seed')
            reports[report_key] = dict(
                task=row['task'], template_sha256=digest,
                report_path=f'acceptance_reports/{report_key}.json',
                report_sha256=proof['report_sha256'], registry_report_path=proof['report_path'],
                evidence_root=proof['evidence_root'], manifest_sha256=report['manifest_sha256'],
                records_sha256=report['records_sha256'],
                annotation_policy_sha256=proof['annotation_policy_sha256'],
                evaluator_sha256=proof['evaluator_sha256'],
            )
        entries.append(dict(row, template_sha256=digest, report_key=report_key))
    required = {task: sorted(hashes) for task, hashes in sorted(required.items())}
    readiness = registry_readiness(registry_root, required)
    if not readiness['ready']:
        raise ValueError('Selected template evidence failed independent reassessment')
    refinement = refinement_readiness(registry_root, required)
    if (require_refinement and not refinement['required']) or not refinement['ready']:
        raise ValueError('Independent constraint refinement evidence is not ready; publication withheld')
    return dict(
        publisher_version=PUBLISHER_VERSION, acceptance_version=ACCEPTANCE_VERSION,
        source_root=str(Path(source_root).resolve()), registry_root=str(Path(registry_root).resolve()),
        source_manifest_sha256=json_hash(manifest), tasks=manifest['tasks'], per_task=EPISODES_PER_TASK,
        annotation_policy_sha256=annotation_policy_fingerprint(), evaluator_sha256=evaluator_fingerprint(),
        required_templates=required, template_reports=reports, refinement_gate=refinement, files=entries,
    )


def episode_proof(plan, row):
    report = plan['template_reports'][row['report_key']]
    return dict(
        version=ACCEPTANCE_VERSION, publisher_version=PUBLISHER_VERSION,
        task=row['task'], source_relative_path=row['path'], source_file_sha256=row['source_sha256'],
        template_sha256=row['template_sha256'], report_sha256=report['report_sha256'],
        acceptance_report=report['report_path'], evidence_root=report['evidence_root'],
        manifest_sha256=report['manifest_sha256'], records_sha256=report['records_sha256'],
        annotation_policy_sha256=plan['annotation_policy_sha256'], evaluator_sha256=plan['evaluator_sha256'],
    )


def check_file(root, row):
    path = contained_path(root, row['path'])
    if not path.is_file() or path.is_symlink() or file_hash(path) != row['sha256']:
        raise ValueError(f'Published episode changed or missing: {row["path"]}')
    with h5py.File(path, 'r') as handle:
        actual = json.loads(handle[PROOF_KEY][()])
        if actual != row['acceptance'] or json_hash(actual) != row['proof_sha256']:
            raise ValueError(f'Embedded acceptance proof mismatch: {row["path"]}')
        meta = json.loads(handle.attrs['robotwin_expert_provenance'])
        if (meta['task'] != row['task'] or meta['seed'] != row['seed']
                or meta['initial_scene_sha256'] != row['initial_scene_sha256']):
            raise ValueError('Published source identity changed')
        read_embedded(handle)


def validate_publication(root, *, require_refinement=False):
    """Read-only full file/proof validation, suitable for the training launch gate."""
    root = Path(root).resolve()
    plan = read_json(root/'publication_plan.json')
    manifest = read_json(root/'acceptance_manifest.json')
    snapshot = read_json(root/'snapshot_complete.json')
    plan_hash = json_hash(plan)
    if (manifest.get('version') != ACCEPTANCE_VERSION or manifest.get('complete') is not True
            or manifest.get('plan_sha256') != plan_hash or snapshot.get('plan_sha256') != plan_hash
            or snapshot.get('acceptance_manifest_sha256') != file_hash(root/'acceptance_manifest.json')
            or manifest.get('files') != snapshot.get('files')):
        raise ValueError('Incomplete or mismatched publication manifest')
    if (plan['annotation_policy_sha256'] != annotation_policy_fingerprint()
            or plan['evaluator_sha256'] != evaluator_fingerprint()):
        raise ValueError('Publication annotation/evaluator code changed')
    refinement = plan.get('refinement_gate', dict(required=False))
    if manifest.get('refinement_gate', dict(required=False)) != refinement:
        raise ValueError('Published refinement requirement differs from its frozen plan')
    validate_frozen_gate(refinement, required_templates=plan['required_templates'], required=require_refinement)
    files = manifest['files']
    if (len(plan['tasks']) != EXPECTED_TASKS or len(set(plan['tasks'])) != EXPECTED_TASKS
            or plan['per_task'] != EPISODES_PER_TASK
            or len(files) != EXPECTED_TASKS*EPISODES_PER_TASK
            or len(plan['files']) != len(files)):
        raise ValueError('Incomplete published task/episode quota')
    expected_paths = {f'{task}/aloha-agilex_clean_50/data/episode{i}.hdf5'
                      for task in plan['tasks'] for i in range(EPISODES_PER_TASK)}
    actual_paths = {str(p.relative_to(root)) for p in root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5')}
    if (len({r['path'] for r in files}) != len(files)
            or {r['path'] for r in files} != {r['path'] for r in plan['files']}
            or {r['path'] for r in files} != expected_paths or actual_paths != expected_paths):
        raise ValueError('Published inventory differs from its frozen plan')
    expected = {row['path']: row for row in plan['files']}
    for row in files:
        if (any(row.get(k) != v for k, v in expected[row['path']].items())
                or row['acceptance'] != episode_proof(plan, expected[row['path']])):
            raise ValueError('Episode proof differs from frozen selection')
        check_file(root, row)
    for report in plan['template_reports'].values():
        path = contained_path(root, report['report_path'])
        if file_hash(path) != report['report_sha256']:
            raise ValueError('Archived acceptance report changed')
        value = read_json(path)
        if (not value.get('ready') or value.get('template_sha256') != report['template_sha256']
                or value.get('task') != report['task'] or value.get('version') != ACCEPTANCE_VERSION
                or value.get('manifest_sha256') != report['manifest_sha256']
                or value.get('records_sha256') != report['records_sha256']
                or value.get('annotation_policy_sha256') != plan['annotation_policy_sha256']
                or value.get('evaluator_sha256') != plan['evaluator_sha256']):
            raise ValueError('Archived report is not accepted for its template')
        fresh = assess_evidence(report['evidence_root'], protocol=value['protocol'],
                                expected_task=report['task'], expected_template_sha256=report['template_sha256'])
        if not derived_values_equal(fresh, value):
            raise ValueError('Archived report no longer matches its raw acceptance evidence')
    batch = read_json(root/'batch_manifest.json')
    if (batch.get('tasks') != plan['tasks'] or batch.get('per_task') != EPISODES_PER_TASK
            or batch.get('collection') != 'clean' or batch.get('running') or batch.get('pending_tasks')
            or batch.get('failed_tasks')):
        raise ValueError('Published batch is not complete')
    counts = {}
    for task in plan['tasks']:
        selected = [row for row in files if row['task'] == task]
        paths = list((root/task/'aloha-agilex_clean_50/data').glob('episode*.hdf5'))
        receipts = list((root/task/'aloha-agilex_clean_50/annotation_queue').glob('episode*.json'))
        if (len(selected) != EPISODES_PER_TASK or len(paths) != EPISODES_PER_TASK
                or len(receipts) != EPISODES_PER_TASK
                or len({r['seed'] for r in selected}) != EPISODES_PER_TASK
                or len({r['initial_scene_sha256'] for r in selected}) != EPISODES_PER_TASK):
            raise ValueError(f'Incomplete or duplicated published task: {task}')
        for row in selected:
            receipt = read_json(root/task/'aloha-agilex_clean_50/annotation_queue'/(Path(row['path']).stem+'.json'))
            if (receipt.get('annotation_status') != 'done' or receipt.get('task') != task
                    or receipt.get('seed') != row['seed']
                    or receipt.get('initial_scene_sha256') != row['initial_scene_sha256']
                    or receipt.get('annotation_result', {}).get('template_sha256') != row['template_sha256']
                    or receipt.get('annotation_result', {}).get('acceptance_report_sha256') != row['acceptance']['report_sha256']):
                raise ValueError('Annotation queue receipt does not match its published episode')
        counts[task] = len(selected)
    return dict(ready=True, version=ACCEPTANCE_VERSION, episodes=len(files), per_task=counts,
                source=plan['source_root'], plan_sha256=plan_hash,
                template_report_hashes={k: v['report_sha256'] for k, v in plan['template_reports'].items()})


def publish(source_root, registry_root, output, *, require_refinement=False):
    source_root, registry_root, output = map(lambda p: Path(p).resolve(), (source_root, registry_root, output))
    if (source_root == output or source_root in output.parents or output in source_root.parents
            or registry_root == output or registry_root in output.parents or output in registry_root.parents):
        raise ValueError('Output must be separate from source data and acceptance registry')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name('.'+output.name+'.publishing')
    with output.with_name('.'+output.name+'.publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Selection and all acceptance checks finish before any new data copy.
        plan = select_plan(source_root, registry_root, require_refinement=require_refinement)
        if output.exists():
            if read_json(output/'publication_plan.json') != plan:
                raise ValueError('Published source/template/proof changed; use a new output version')
            return validate_publication(output, require_refinement=require_refinement)
        if staging.exists():
            if staging.is_symlink():
                raise ValueError('Staging directory must not be a symlink')
            if read_json(staging/'publication_plan.json') != plan:
                raise ValueError('Staged source/template/proof changed; use a new output version')
        else:
            staging.mkdir()
            atomic_json(staging/'publication_plan.json', plan)
        for report in plan['template_reports'].values():
            target = contained_path(staging, report['report_path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if file_hash(target) != report['report_sha256']:
                    raise ValueError('Staged acceptance report changed')
            else:
                shutil.copy2(report['registry_report_path'], target)
                if file_hash(target) != report['report_sha256']:
                    raise ValueError('Acceptance report changed while copying')
        rows = []
        for source in plan['files']:
            destination = contained_path(staging, source['path'])
            receipt_path = contained_path(staging, Path('publication_receipts')/source['task']/(destination.stem+'.json'))
            if receipt_path.exists():
                row = read_json(receipt_path)
                if row['acceptance'] != episode_proof(plan, source):
                    raise ValueError('Staged episode belongs to a different proof')
                check_file(staging, row)
            else:
                original = source_root/source['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                # An interrupted file with no committed receipt is safe to
                # replace: staging contains no user source or published data.
                fd, temporary_name = tempfile.mkstemp(prefix='.'+destination.name+'.copy.', dir=destination.parent)
                os.close(fd)
                temporary = Path(temporary_name)
                try:
                    shutil.copy2(original, temporary)
                    if os.path.samefile(original, temporary) or file_hash(temporary) != source['source_sha256']:
                        raise ValueError('Source changed or copy shares its inode')
                    # Replace the directory entry instead of opening an old
                    # partial file, which could have become a hardlink/symlink.
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                matched = match(registry_root, source['task'], original)
                if matched is None or template_hash(matched[0]['template']) != source['template_sha256']:
                    raise ValueError('Accepted template changed after planning')
                entry, constraints, _ = matched
                if entry['acceptance']['report_sha256'] != plan['template_reports'][source['report_key']]['report_sha256']:
                    raise ValueError('Acceptance proof changed after planning')
                result = embed_review(destination, dict(source_sha256=source['source_sha256'], constraints=constraints),
                                      agent_run='acceptance-v2:'+source['template_sha256'])
                proof = episode_proof(plan, source)
                with h5py.File(destination, 'r+') as handle:
                    if PROOF_KEY in handle:
                        del handle[PROOF_KEY]
                    handle.create_dataset(PROOF_KEY, data=json.dumps(proof, sort_keys=True), dtype=h5py.string_dtype())
                if file_hash(original) != source['source_sha256']:
                    raise ValueError('Source changed during publication')
                row = dict(source, sha256=file_hash(destination), bytes=destination.stat().st_size,
                           proof_sha256=json_hash(proof), acceptance=proof, revision=result['revision'])
                atomic_json(receipt_path, row)
            rows.append(row)
            queue = contained_path(staging, Path(source['task'])/'aloha-agilex_clean_50/annotation_queue'/(destination.stem+'.json'))
            atomic_json(queue, dict(task=source['task'], index=int(destination.stem.removeprefix('episode')),
                                   seed=source['seed'], initial_scene_sha256=source['initial_scene_sha256'],
                                   collection=0, task_config='demo_clean', native_success=True,
                                   path=str(output/source['path']), annotation_status='done',
                                   annotation_result=dict(template_sha256=source['template_sha256'],
                                                          acceptance_report_sha256=row['acceptance']['report_sha256'],
                                                          revision=row['revision'])))
        # Recheck current evidence once after all files, without modifying it.
        if not registry_readiness(registry_root, plan['required_templates'])['ready']:
            raise ValueError('Acceptance evidence changed before publication')
        batch = dict(tasks=plan['tasks'], per_task=EPISODES_PER_TASK, total_target=len(rows), collection='clean',
                     status='published', running={}, pending_tasks=[], failed_tasks={},
                     recorded_tasks=plan['tasks'], source_root=str(source_root), updated_unix=time.time())
        atomic_json(staging/'batch_manifest.json', batch)
        atomic_json(staging/'acceptance_manifest.json', dict(version=ACCEPTANCE_VERSION, complete=True,
                    plan_sha256=json_hash(plan), source_root=str(source_root), refinement_gate=plan['refinement_gate'], files=rows))
        atomic_json(staging/'snapshot_complete.json', dict(source=str(source_root), plan_sha256=json_hash(plan),
                    acceptance_manifest_sha256=file_hash(staging/'acceptance_manifest.json'), files=rows))
        validated = validate_publication(staging, require_refinement=require_refinement)
        os.replace(staging, output)
        return validated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--registry-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--require-refinement', action='store_true',
                        help='Require independent model-aware holdout evidence even if the registry marker is missing')
    args = parser.parse_args()
    if args.validate_only:
        result = validate_publication(args.output, require_refinement=args.require_refinement)
    else:
        if args.source_root is None or args.registry_root is None:
            parser.error('--source-root and --registry-root are required for publication')
        result = publish(args.source_root, args.registry_root, args.output, require_refinement=args.require_refinement)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
