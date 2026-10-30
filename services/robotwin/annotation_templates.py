"""Task template registry with explicit counterfactual acceptance evidence.

Reference consistency is necessary but never activates a new template by itself.
Old active entries remain readable only through an explicit legacy query.
"""
import json
from pathlib import Path
import time

import h5py

from services.robotwin.collection import atomic_json
from services.robotwin.programmatic_annotation import OutsideTemplate, instantiate, template_hash, validate_template
from services.robotwin.acceptance import (
    ACCEPTANCE_VERSION, annotation_policy_fingerprint, assess_evidence, evaluator_fingerprint,
    derived_values_equal, file_hash, json_hash,
)


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
                          'agent_run': str(agent_run), 'created_unix': time.time(), 'validation': [],
                          'acceptance_version': ACCEPTANCE_VERSION})
    refresh(root, template['task'])
    return json.loads(path.read_text())


def refresh(root, task, *, reference_root=None):
    directory = Path(reference_root or root)/task/'aloha-agilex_clean_50'
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
                entry.update(reference_consistent=True)
                break
        entry['validation'] = audit
        if entry['status'] != 'rejected':
            ready, reason = acceptance_ready(entry)
            entry.update(status='active' if ready else 'candidate', acceptance_blocker=reason)
            if ready:
                entry.setdefault('activated_unix', time.time())
        atomic_json(registry, entry)


def match(root, task, episode, *, include_candidates=False, allow_legacy=False):
    statuses = ('active', 'candidate') if include_candidates else ('active',)
    candidates = [e for _, e in entries(root, task) if e['status'] in statuses
                  and (include_candidates or acceptance_ready(e)[0] or allow_legacy)]
    if not candidates: return None
    with h5py.File(episode, 'r') as h:
        for entry in candidates:
            try:
                constraints, audit = instantiate(h, entry['template'])
                return entry, constraints, audit
            except OutsideTemplate:
                continue
    return None


def acceptance_ready(entry, *, verify_artifacts=False):
    proof = entry.get('acceptance')
    if not proof or proof.get('version') != ACCEPTANCE_VERSION:
        return False, 'counterfactual_evidence_missing'
    if proof.get('annotation_policy_sha256') != annotation_policy_fingerprint():
        return False, 'annotation_policy_changed'
    if proof.get('evaluator_sha256') != evaluator_fingerprint():
        return False, 'acceptance_evaluator_changed'
    try:
        path = Path(proof['report_path'])
        if file_hash(path) != proof['report_sha256']:
            return False, 'acceptance_report_changed'
        report = json.loads(path.read_text())
        if (not report.get('ready') or report.get('version') != ACCEPTANCE_VERSION
                or report.get('template_sha256') != template_hash(entry['template'])
                or report.get('task') != entry['template']['task']
                or report.get('annotation_policy_sha256') != proof['annotation_policy_sha256']
                or report.get('evaluator_sha256') != proof['evaluator_sha256']):
            return False, 'acceptance_report_not_ready_or_mismatched'
        manifest = Path(proof['evidence_root'])/'experiment_manifest.json'
        records = Path(proof['evidence_root'])/'trials.jsonl'
        if file_hash(manifest) != report['manifest_sha256'] or file_hash(records) != report['records_sha256']:
            return False, 'acceptance_evidence_changed'
        if verify_artifacts:
            fresh = assess_evidence(proof['evidence_root'], protocol=report['protocol'],
                                    expected_task=entry['template']['task'],
                                    expected_template_sha256=template_hash(entry['template']))
            if not derived_values_equal(fresh, report):
                return False, 'acceptance_raw_artifacts_changed'
    except (KeyError, ValueError, FileNotFoundError):
        return False, 'acceptance_artifacts_missing_or_invalid'
    return True, None


def attach_evidence(root, task, template_sha256, evidence_root, *, protocol=None, reference_root=None):
    """Assess one template's sealed trials, then attach a receipt to a new registry.

    Only this registry and its report file are written. ``reference_root`` can
    point at a read-only source dataset to ensure the manifest excludes every
    training seed/scene, while evidence and the registry live elsewhere.
    """
    path = Path(root)/'annotation_templates'/task/(template_sha256+'.json')
    entry = json.loads(path.read_text())
    if template_hash(entry['template']) != template_sha256 or entry['template']['task'] != task:
        raise ValueError('Registry identity mismatch')
    evidence_root = Path(evidence_root).resolve()
    report = assess_evidence(evidence_root, protocol=protocol, expected_task=task,
                             expected_template_sha256=template_sha256)
    manifest = json.loads((evidence_root/'experiment_manifest.json').read_text())
    if reference_root is not None:
        seeds, scenes = training_identities(reference_root, task)
        if (not seeds <= set(manifest['excluded_training_seeds'])
                or not scenes <= set(manifest['excluded_training_scene_sha256'])):
            raise ValueError('Evidence omitted training seeds/scenes from the exclusion inventory')
    report_path = Path(root)/'acceptance_reports'/task/(template_sha256+'.json')
    atomic_json(report_path, report)
    entry.update(status='active' if report['ready'] else 'candidate', acceptance_version=ACCEPTANCE_VERSION,
                 acceptance=dict(version=ACCEPTANCE_VERSION, evidence_root=str(evidence_root),
                                 report_path=str(report_path.resolve()), report_sha256=file_hash(report_path),
                                 annotation_policy_sha256=report['annotation_policy_sha256'],
                                 evaluator_sha256=report['evaluator_sha256']),
                 acceptance_blocker=None if report['ready'] else report['reasons'])
    if report['ready']:
        entry['activated_unix'] = time.time()
    atomic_json(path, entry)
    return report


def training_identities(root, task):
    directory = Path(root)/task/'aloha-agilex_clean_50'
    paths = sorted({*directory.glob('data/episode*.hdf5'), *directory.glob('raw/episode*.hdf5')})
    if not paths:
        raise ValueError('Cannot verify training exclusions without source recordings')
    seeds, scenes = set(), set()
    for path in paths:
        with h5py.File(path, 'r') as handle:
            meta = json.loads(handle.attrs['robotwin_expert_provenance'])
            if meta['task'] != task:
                raise ValueError('Source dataset task identity mismatch')
            seeds.add(meta['seed']); scenes.add(meta['initial_scene_sha256'])
    return seeds, scenes


def registry_readiness(root, required_templates=None, *, verify_artifacts=True):
    """Read-only report over every template actually selected for publication.

    required_templates is an explicit {task: [template_sha256, ...]} mapping.
    When omitted, all non-rejected registry entries are checked; callers must
    not infer episode coverage merely from one passing template per task.
    """
    if required_templates is None:
        required_templates = {}
        for path in sorted((Path(root)/'annotation_templates').glob('*/*.json')):
            entry = json.loads(path.read_text())
            if entry.get('status') != 'rejected':
                required_templates.setdefault(path.parent.name, []).append(path.stem)
    rows = []
    for task, hashes in sorted(required_templates.items()):
        for digest in sorted(set(hashes)):
            path = Path(root)/'annotation_templates'/task/(digest+'.json')
            try:
                entry = json.loads(path.read_text())
                if template_hash(entry['template']) != digest or entry['template']['task'] != task:
                    raise ValueError('registry identity mismatch')
                ready, reason = acceptance_ready(entry, verify_artifacts=verify_artifacts)
                ready = ready and entry['status'] == 'active'
                rows.append(dict(task=task, template_sha256=digest, ready=ready,
                                 reason=reason if not ready else None,
                                 report_sha256=entry.get('acceptance', {}).get('report_sha256')))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                rows.append(dict(task=task, template_sha256=digest, ready=False, reason=str(exc)))
    return dict(version=ACCEPTANCE_VERSION, ready=bool(rows) and all(r['ready'] for r in rows),
                templates=rows, required_templates_sha256=json_hash(required_templates))
