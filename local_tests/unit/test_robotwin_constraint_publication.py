"""Accepted-only publication preserves demonstrations and fails closed on resume."""
import json
import os

import h5py
import numpy as np
import pytest

from scripts import publish_robotwin_constraint_dataset as publisher
from services.robotwin import annotation_templates as registry
from services.robotwin.collection import atomic_json
from services.robotwin.constraint_schema import read_embedded
from services.robotwin.programmatic_annotation import template_hash
from test_robotwin_acceptance import evidence  # reusable synthetic native-evidence fixture
from test_robotwin_programmatic_annotation import episode


@pytest.fixture
def publication(tmp_path, monkeypatch, evidence):
    # Quota mechanics run on a tiny fixture; CLI production constants remain 50 x 50.
    monkeypatch.setattr(publisher, 'EXPECTED_TASKS', 1)
    monkeypatch.setattr(publisher, 'EPISODES_PER_TASK', 3)
    source, accepted, output = (tmp_path/name for name in ('source', 'registry', 'published'))
    paths = []
    for seed, scene in enumerate(('train-a', 'train-b', 'train-c')):
        path = episode(source/'adjust_bottle/aloha-agilex_clean_50/data'/f'episode{seed}.hdf5',
                       seed=seed, fingerprint=scene)
        with h5py.File(path, 'r+') as handle:
            meta = json.loads(handle.attrs['robotwin_expert_provenance'])
            meta.update(collection=0, task_config='demo_clean', native_success=True)
            handle.attrs['robotwin_expert_provenance'] = json.dumps(meta)
            handle.attrs['instruction'] = 'Keep the original demonstration instruction'
            handle.create_dataset('observation/head_camera/rgb', data=np.arange(24, dtype=np.uint8).reshape(2, 3, 4))
            handle.create_dataset('joint_action/vector', data=np.arange(28).reshape(2, 14))
            handle.create_dataset('initial_scene', data=json.dumps({'seed': seed, 'object_pose': [0, seed, 1]}))
        paths.append(path)
    atomic_json(source/'batch_manifest.json', dict(tasks=['adjust_bottle'], per_task=3, collection='clean',
                running={}, pending_tasks=[], failed_tasks={}))
    evidence_root, manifest, _ = evidence
    registry.install(accepted, manifest['template'], source_episode=paths[0], agent_run='synthetic-test')
    digest = template_hash(manifest['template'])
    report = registry.attach_evidence(accepted, 'adjust_bottle', digest, evidence_root, reference_root=source)
    assert report['ready']
    return dict(source=source, registry=accepted, output=output, paths=paths,
                evidence=evidence_root, digest=digest)


def run(publication):
    return publisher.publish(publication['source'], publication['registry'], publication['output'])


def test_publish_preserves_sources_actions_images_and_scene(publication):
    before = {p: publisher.file_hash(p) for p in publication['paths']}
    result = run(publication)
    assert result['ready'] and result['episodes'] == 3
    assert result['per_task'] == {'adjust_bottle': 3}
    for source in publication['paths']:
        dest = publication['output']/source.relative_to(publication['source'])
        assert publisher.file_hash(source) == before[source]
        assert not os.path.samefile(source, dest)
        with h5py.File(source, 'r') as old, h5py.File(dest, 'r') as new:
            assert dict(old.attrs) == dict(new.attrs)
            for key in ('observation/head_camera/rgb', 'joint_action/vector', 'initial_scene',
                        'endpose/left_endpose', 'endpose/right_endpose',
                        'endpose/left_gripper', 'endpose/right_gripper'):
                np.testing.assert_array_equal(old[key][()], new[key][()])
            assert read_embedded(old)['constraints'] == []
            assert read_embedded(new)['constraints']
            proof = json.loads(new[publisher.PROOF_KEY][()])
            assert proof['source_file_sha256'] == before[source]
            assert proof['template_sha256'] == publication['digest']
            # Proof is outside strict constraint-bundle provenance.
            assert 'acceptance' not in read_embedded(new)['provenance']
    assert publisher.validate_publication(publication['output']) == result
    assert not publication['output'].with_name('.published.publishing').exists()
    assert len(list(publication['output'].rglob('*.hdf5'))) == 3


def test_only_counterfactually_accepted_templates_can_publish(publication):
    path = next((publication['registry']/'annotation_templates').glob('*/*.json'))
    entry = publisher.read_json(path)
    # Old legacy entries may claim active but do not carry acceptance-v2 proof.
    del entry['acceptance']
    entry['status'] = 'active'
    atomic_json(path, entry)
    with pytest.raises(ValueError, match='No accepted matching template'):
        run(publication)
    assert not publication['output'].exists()
    assert not publication['output'].with_name('.published.publishing').exists()


def test_refinement_marker_prevents_pilot_only_publication(publication):
    from services.robotwin.constraint_refinement import DEFAULT_PROTOCOL
    atomic_json(publication['registry']/'refinement_requirement.json', dict(version=1, protocol=DEFAULT_PROTOCOL))
    with pytest.raises(ValueError, match='refinement evidence is not ready'):
        run(publication)
    assert not publication['output'].exists()
    assert not publication['output'].with_name('.published.publishing').exists()


def test_training_cannot_silently_omit_required_refinement(publication):
    with pytest.raises(ValueError, match='refinement evidence is not ready'):
        publisher.publish(publication['source'], publication['registry'], publication['output'], require_refinement=True)
    run(publication)  # Historical pilot-only artifact remains explicitly distinguishable.
    with pytest.raises(ValueError, match='refinement evidence is required'):
        publisher.validate_publication(publication['output'], require_refinement=True)


def test_existing_manual_annotation_does_not_bypass_new_template(publication):
    # All fixture source constraints are empty/manual; every episode still needs
    # a matching, accepted template. A second event changes this episode's signature.
    with h5py.File(publication['paths'][-1], 'r+') as handle:
        handle['endpose/left_gripper'][-1] = 1.
    with pytest.raises(ValueError, match='No accepted matching template'):
        run(publication)
    assert not publication['output'].exists()


def test_completed_resume_verifies_hash_and_never_recopies(publication, monkeypatch):
    first = run(publication)
    monkeypatch.setattr(publisher.shutil, 'copy2', lambda *args, **kwargs: pytest.fail('resume recopied data'))
    assert run(publication) == first
    target = publication['output']/publication['paths'][0].relative_to(publication['source'])
    with h5py.File(target, 'r+') as handle:
        handle['joint_action/vector'][0, 0] += 1
    with pytest.raises(ValueError, match='Published episode changed'):
        run(publication)


def test_interrupted_copy_resumes_only_committed_matching_files(publication, monkeypatch):
    real_embed = publisher.embed_review
    count = 0
    def fail_third(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise RuntimeError('simulated interruption')
        return real_embed(*args, **kwargs)
    monkeypatch.setattr(publisher, 'embed_review', fail_third)
    with pytest.raises(RuntimeError, match='interruption'):
        run(publication)
    assert not publication['output'].exists()
    staging = publication['output'].with_name('.published.publishing')
    assert len(list((staging/'publication_receipts').rglob('*.json'))) == 2
    # Even a hostile/stale uncommitted hardlink must be replaced, never opened
    # for copying or annotation; its original inode must remain unchanged.
    unfinished = staging/publication['paths'][2].relative_to(publication['source'])
    unfinished.unlink()
    os.link(publication['paths'][2], unfinished)
    original_hash = publisher.file_hash(publication['paths'][2])
    count = 0
    def record_embed(*args, **kwargs):
        nonlocal count
        count += 1
        return real_embed(*args, **kwargs)
    monkeypatch.setattr(publisher, 'embed_review', record_embed)
    assert run(publication)['ready']
    assert count == 1
    assert publisher.file_hash(publication['paths'][2]) == original_hash


def test_source_change_refuses_mixed_output(publication):
    run(publication)
    with h5py.File(publication['paths'][0], 'r+') as handle:
        handle.attrs['instruction'] = 'A changed original'
    with pytest.raises(ValueError, match='source/template/proof changed'):
        run(publication)


def test_missing_task_cannot_publish_partial_dataset(publication, monkeypatch):
    monkeypatch.setattr(publisher, 'EXPECTED_TASKS', 2)
    path = publication['source']/'batch_manifest.json'
    manifest = publisher.read_json(path)
    manifest['tasks'].append('missing_task')
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match='episode quota: missing_task'):
        run(publication)
    assert not publication['output'].exists()
    assert not publication['output'].with_name('.published.publishing').exists()


@pytest.mark.parametrize('duplicate', ['seed', 'initial_scene_sha256'])
def test_repeated_initial_states_block_publication(publication, duplicate):
    with h5py.File(publication['paths'][1], 'r+') as handle:
        meta = json.loads(handle.attrs['robotwin_expert_provenance'])
        meta[duplicate] = 0 if duplicate == 'seed' else 'train-a'
        handle.attrs['robotwin_expert_provenance'] = json.dumps(meta)
    with pytest.raises(ValueError, match='Duplicate initial state'):
        run(publication)


def test_mutated_acceptance_artifact_blocks_training_validation(publication):
    run(publication)
    trajectory = next((publication['evidence']/'trajectories').glob('*.json'))
    atomic_json(trajectory, {'actual': {}})
    with pytest.raises(ValueError, match='raw acceptance evidence'):
        publisher.validate_publication(publication['output'])


@pytest.mark.parametrize('delta,accepted', [(1e-13, True), (1e-5, False)])
def test_recomputed_report_allows_only_numerical_roundoff(publication, monkeypatch, delta, accepted):
    run(publication)
    assess = publisher.assess_evidence
    def recomputed(*args, **kwargs):
        report = assess(*args, **kwargs)
        report['trials'][0]['max_violation'] += delta
        return report
    monkeypatch.setattr(publisher, 'assess_evidence', recomputed)
    if accepted:
        assert publisher.validate_publication(publication['output'])['ready']
    else:
        with pytest.raises(ValueError, match='raw acceptance evidence'):
            publisher.validate_publication(publication['output'])


def test_source_cannot_be_used_as_output(publication):
    with pytest.raises(ValueError, match='separate from source'):
        publisher.publish(publication['source'], publication['registry'], publication['source'])


def test_missing_done_receipt_blocks_training_validation(publication):
    run(publication)
    receipt = next(publication['output'].glob('*/aloha-agilex_clean_50/annotation_queue/*.json'))
    receipt.unlink()
    with pytest.raises(ValueError, match='Incomplete or duplicated'):
        publisher.validate_publication(publication['output'])
