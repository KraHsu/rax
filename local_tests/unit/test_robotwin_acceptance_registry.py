"""Publication requires proof for every selected template, including manual paths."""
import copy

import h5py
import pytest

from services.robotwin.annotation_templates import (acceptance_ready, attach_evidence,
                                                   match, registry_readiness)
from services.robotwin.collection import atomic_json
from services.robotwin.programmatic_annotation import instantiate, template_hash
from test_robotwin_acceptance import template

pytest_plugins = ('test_robotwin_acceptance',)
from test_robotwin_programmatic_annotation import episode


def register(root, t, *, status='candidate'):
    path = root/'annotation_templates'/t['task']/(template_hash(t)+'.json')
    atomic_json(path, dict(template=t, status=status, source_episode='legacy-reference', validation=[]))
    return path


def test_legacy_active_requires_explicit_readonly_opt_in(tmp_path):
    t = template()
    registry = register(tmp_path, t, status='active')
    path = episode(tmp_path/'source.hdf5')
    before = registry.read_bytes()
    assert match(tmp_path, t['task'], path) is None
    assert match(tmp_path, t['task'], path, allow_legacy=True) is not None
    assert registry.read_bytes() == before
    assert not registry_readiness(tmp_path)['ready']


def test_attach_ready_evidence_and_match_are_template_specific(evidence, tmp_path):
    evidence_root, manifest, _ = evidence
    registry_root = tmp_path/'registry'
    t = manifest['template']
    register(registry_root, t)
    report = attach_evidence(registry_root, t['task'], template_hash(t), evidence_root)
    assert report['ready']
    path = episode(tmp_path/'source.hdf5')
    entry, _, _ = match(registry_root, t['task'], path)
    assert acceptance_ready(entry, verify_artifacts=True)[0]
    assert registry_readiness(registry_root, {t['task']: [template_hash(t)]})['ready']
    other = copy.deepcopy(t)
    other['rationale'] += ' another signature template'
    register(registry_root, other)
    assert not registry_readiness(registry_root)['ready']


def test_report_artifacts_are_revalidated_before_publication(evidence, tmp_path):
    evidence_root, manifest, records = evidence
    registry_root = tmp_path/'registry'
    t = manifest['template']
    register(registry_root, t)
    attach_evidence(registry_root, t['task'], template_hash(t), evidence_root)
    artifact = evidence_root/records[0]['trajectory_path']
    artifact.write_text('{}')
    assert not registry_readiness(registry_root)['ready']


def test_all_training_seed_and_scene_ids_must_be_excluded(evidence, tmp_path):
    evidence_root, manifest, _ = evidence
    registry_root = tmp_path/'registry'
    t = manifest['template']
    register(registry_root, t)
    source = tmp_path/'source'
    episode(source/t['task']/'aloha-agilex_clean_50/data/episode0.hdf5', seed=9, fingerprint='not-in-manifest')
    with pytest.raises(ValueError, match='omitted training'):
        attach_evidence(registry_root, t['task'], template_hash(t), evidence_root, reference_root=source)


def test_manual_completion_cannot_bypass_new_gate(tmp_path):
    from scripts.robotwin_annotation_queue import complete
    from services.robotwin.annotation import digest
    t = template()
    root = tmp_path/'dataset'
    path = episode(root/t['task']/'aloha-agilex_clean_50/raw/episode0.hdf5')
    with h5py.File(path) as handle:
        cs, _ = instantiate(handle, t)
    receipt = path.parent.parent/'annotation_queue/episode0.json'
    atomic_json(receipt, dict(task=t['task'], path=str(path), claim_id='claim', annotation_status='processing'))
    atomic_json(root/'annotation_control.json', dict(acceptance_version=2))
    register(root, t, status='active')
    original = path.read_bytes()
    with pytest.raises(ValueError, match='Counterfactual acceptance required'):
        complete(receipt, 'claim', dict(source_sha256=digest(path), constraints=cs), 'agent-test')
    assert path.read_bytes() == original and not (path.parent.parent/'data/episode0.hdf5').exists()
