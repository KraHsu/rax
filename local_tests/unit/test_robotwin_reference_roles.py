"""Role identity is compiled from a fixed reference, never a model prediction."""
import copy
import json

import h5py
import numpy as np
import pytest

from services.robotwin.annotation import digest, embed_review
from services.robotwin.constraint_schema import read_embedded, validate_bundle
from services.robotwin.programmatic_annotation import OutsideTemplate, instantiate, template_hash, validate_template
from test_robotwin_programmatic_annotation import episode, template


def role_template():
    t = template()
    t['signatures'] = [['close', 'close,open']]
    t['reference_roles'] = [
        dict(role='basket_carrier', source='reference_event_signature', event_signature='close',
             rationale='Native carrier remains closed after the other hand releases the placed object.'),
        dict(role='object_placer', source='reference_event_signature', event_signature='close,open',
             rationale='Native object hand closes and releases the object before the carrier lifts.'),
    ]
    for rule in t['rules'][-2:]:
        rule['subjects'] = 'role:basket_carrier'
    return t


def two_arm_episode(path, carrier='right'):
    episode(path)
    with h5py.File(path, 'r+') as h:
        for arm in ('left', 'right'):
            pose = h[f'endpose/{arm}_endpose'][()]
            pose[:, 2] = .8 + np.linspace(0, .1 if arm == carrier else .02, len(pose))
            grip = np.ones(len(pose)); grip[2:] = 0
            if arm != carrier:
                grip[-2:] = 1
            h[f'endpose/{arm}_endpose'][...] = pose
            h[f'endpose/{arm}_gripper'][...] = grip
    return path


@pytest.mark.parametrize('carrier', ['left', 'right'])
def test_same_template_binds_native_roles_and_preserves_numeric_contract(tmp_path, carrier):
    t = role_template()
    with h5py.File(two_arm_episode(tmp_path/'ep.hdf5', carrier)) as h:
        constraints, audit = instantiate(h, t)
        direct = copy.deepcopy(t); del direct['reference_roles']
        for rule in direct['rules'][-2:]:
            rule['subjects'] = carrier
        expected, _ = instantiate(h, direct)
        # Only provenance differs from an explicitly correct-arm compilation.
        strip = lambda values: [{k: v for k, v in c.items() if k != 'evidence'} for c in values]
        assert strip(constraints) == strip(expected)
        bundle = read_embedded(h); bundle['constraints'] = constraints; bundle['schema_version'] = '1.4'
        validate_bundle(bundle)
        assert all(c['subject'] in ('left_eef', 'right_eef') for c in constraints)
        roles = audit['reference_roles']
        assert roles['bindings']['basket_carrier'] == carrier
        assert roles['bindings']['object_placer'] != carrier
        assert not roles['generated_actions_used']
        assert roles['declarations_sha256'] == template_hash(t['reference_roles'])
        assert audit['coverage']['complete']


@pytest.mark.parametrize('bad', ['duplicate_name', 'missing_declaration', 'unknown_role', 'empty',
                                  'generated_source', 'invalid_signature', 'executable'])
def test_role_declarations_are_strict_data_only(bad):
    t = role_template()
    if bad == 'duplicate_name':
        t['reference_roles'][1]['role'] = 'basket_carrier'
    elif bad == 'missing_declaration':
        del t['reference_roles']
    elif bad == 'unknown_role':
        t['rules'][-1]['subjects'] = 'role:missing'
    elif bad == 'empty':
        t['reference_roles'] = []
    elif bad == 'generated_source':
        t['reference_roles'][0]['source'] = 'predicted_gripper'
    elif bad == 'invalid_signature':
        t['reference_roles'][0]['event_signature'] = 'close,close'
    else:
        t['reference_roles'][0]['python'] = 'pick_arm()'
    with pytest.raises(ValueError):
        validate_template(t)


@pytest.mark.parametrize('failure', ['same_signature', 'missing_match', 'same_arm_two_roles'])
def test_unresolved_roles_fail_even_with_an_allowed_sorted_signature(tmp_path, failure):
    t = role_template(); path = two_arm_episode(tmp_path/'ep.hdf5')
    with h5py.File(path, 'r+') as h:
        if failure == 'same_signature':
            h['endpose/left_gripper'][...] = h['endpose/right_gripper'][()]
            t['signatures'] = [['close', 'close']]
        elif failure == 'missing_match':
            t['reference_roles'][0]['event_signature'] = 'open,close'
        else:
            t['reference_roles'][1]['event_signature'] = 'close'
        with pytest.raises(OutsideTemplate, match='exactly one|same arm'):
            instantiate(h, t)


def test_reference_hash_covers_role_input_and_activation_is_not_recomputed_for_predictions(tmp_path, monkeypatch):
    from services.robotwin import programmatic_annotation as compiler
    from services.robotwin.acceptance import trajectory_violations
    path = two_arm_episode(tmp_path/'ep.hdf5')
    with h5py.File(path, 'r') as h:
        constraints, audit = instantiate(h, role_template())
        reference = {a: dict(pose=h[f'endpose/{a}_endpose'][()].tolist(),
                             gripper=h[f'endpose/{a}_gripper'][()].tolist()) for a in ('left', 'right')}
        bundle = read_embedded(h); bundle.update(schema_version='1.4', constraints=constraints)
        bundle['provenance'].update(split='validation', variant='clean_heldout', source='acceptance_reference_not_for_training')
    candidate = copy.deepcopy(reference)
    candidate['left']['gripper'], candidate['right']['gripper'] = candidate['right']['gripper'], candidate['left']['gripper']
    frozen = copy.deepcopy(constraints)
    monkeypatch.setattr(compiler, 'resolve_reference_roles', lambda *args: pytest.fail('Model scoring must not resolve roles'))
    scored = trajectory_violations(bundle, reference, candidate)
    assert not scored['passed'] and bundle['constraints'] == frozen
    monkeypatch.undo()
    with h5py.File(path, 'r+') as h:
        pose = h['endpose/left_endpose'][()]; pose[:, 0] += .001; h['endpose/left_endpose'][...] = pose
        _, changed = instantiate(h, role_template())
    assert changed['reference_roles']['bindings'] == audit['reference_roles']['bindings']
    assert changed['reference_roles']['reference_sha256'] != audit['reference_roles']['reference_sha256']


@pytest.mark.parametrize('coverage_policy', [None, dict(half_angle_rad=1.2, hold_margin_m=.04, direction_slack_m=.006)])
def test_role_audit_is_recomputed_under_source_lock_and_persisted_in_dataset_history(tmp_path, coverage_policy):
    path = two_arm_episode(tmp_path/'ep.hdf5'); t = role_template()
    if coverage_policy is not None:
        t['coverage_policy'] = coverage_policy
    with h5py.File(path, 'r') as h:
        constraints, audit = instantiate(h, t)
    proposal = dict(source_sha256=digest(path), constraints=constraints)
    embed_review(path, proposal, agent_run='role-test', reference_template=t)
    with h5py.File(path, 'r') as h:
        history = json.loads(h['trajectory_constraints/history'][()])
        stored = history[-1]['programmatic_audit']
        assert stored['reference_roles'] == audit['reference_roles']
        assert stored['template_sha256'] == template_hash(t)
        assert read_embedded(h)['constraints'] == constraints
    before = digest(path)
    changed = copy.deepcopy(t); changed['reference_roles'][0]['rationale'] += ' Changed source claim.'
    with pytest.raises(ValueError, match='differs from the fixed reference'):
        embed_review(path, dict(source_sha256=before, constraints=constraints),
                     agent_run='mismatch', reference_template=changed)
    assert digest(path) == before


def test_templates_without_roles_keep_old_audit_and_annotation_history(tmp_path):
    path = episode(tmp_path/'old.hdf5')
    with h5py.File(path, 'r') as h:
        constraints, audit = instantiate(h, template())
    assert set(audit) == {'template_sha256', 'signature', 'policy', 'reference_audit', 'coverage'}
    embed_review(path, dict(source_sha256=digest(path), constraints=constraints), agent_run='legacy')
    with h5py.File(path, 'r') as h:
        assert 'programmatic_audit' not in json.loads(h['trajectory_constraints/history'][()])[-1]


def test_non_role_template_also_preserves_accepted_coverage_policy_on_embed(tmp_path):
    path = episode(tmp_path/'coverage.hdf5'); t = template()
    t['coverage_policy'] = dict(half_angle_rad=1.2, hold_margin_m=.04)
    with h5py.File(path, 'r') as h:
        constraints, _ = instantiate(h, t)
    embed_review(path, dict(source_sha256=digest(path), constraints=constraints),
                 agent_run='accepted-coverage', reference_template=t)
    with h5py.File(path, 'r') as h:
        assert read_embedded(h)['constraints'] == constraints
        assert 'programmatic_audit' not in json.loads(h['trajectory_constraints/history'][()])[-1]


def test_existing_rsi_cannot_smuggle_subject_or_role_binding_changes():
    from services.robotwin.constraint_refinement import development_summary, propose_revision
    from test_robotwin_constraint_refinement import row
    parent = role_template(); dev = development_summary([row()])
    rule = copy.deepcopy(parent['rules'][-1]); rule['subjects'] = 'left'
    patch = dict(version=1, base_template_sha256=template_hash(parent), reason='Test forbidden role reassignment',
                 evidence_ids=[dev['rows'][0]['trial_id']], operations=[dict(op='replace_rule', rule=rule)])
    with pytest.raises(ValueError, match='activation'):
        propose_revision(parent, patch, dev)
    patch['operations'] = [dict(op='set_reference_roles', reference_roles=parent['reference_roles'])]
    with pytest.raises(ValueError, match='whitelisted'):
        propose_revision(parent, patch, dev)


def test_queue_cannot_use_a_role_template_to_bypass_acceptance(tmp_path, monkeypatch):
    from scripts.robotwin_annotation_queue import complete
    from services.robotwin import annotation_templates
    directory = tmp_path/'adjust_bottle/aloha-agilex_clean_50'
    path = two_arm_episode(directory/'raw/episode0.hdf5')
    receipt = directory/'annotation_queue/episode0.json'; receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(claim_id='lease', annotation_status='processing', task='adjust_bottle', path=str(path))))
    with h5py.File(path, 'r') as h:
        constraints, _ = instantiate(h, role_template())
    monkeypatch.setattr(annotation_templates, 'match', lambda *args: None)
    before = digest(path)
    with pytest.raises(ValueError, match='Counterfactual acceptance required'):
        complete(receipt, 'lease', dict(source_sha256=before, constraints=constraints),
                 'role-test', reference_template=role_template())
    assert path.exists() and digest(path) == before
    assert not (directory/'data/episode0.hdf5').exists()


def test_accepted_queue_preserves_role_audit_and_coverage_policy(tmp_path, monkeypatch):
    from scripts.robotwin_annotation_queue import complete
    from services.robotwin import annotation_templates
    directory = tmp_path/'adjust_bottle/aloha-agilex_clean_50'
    path = two_arm_episode(directory/'raw/episode0.hdf5'); t = role_template()
    t['coverage_policy'] = dict(hold_margin_m=.04)
    receipt = directory/'annotation_queue/episode0.json'; receipt.parent.mkdir()
    receipt.write_text(json.dumps(dict(claim_id='lease', annotation_status='processing', task='adjust_bottle', path=str(path))))
    with h5py.File(path, 'r') as h:
        constraints, audit = instantiate(h, t)
    # Acceptance verification is covered by registry tests; this exercises the
    # actual post-match queue/atomic annotation/publication plumbing.
    entry = dict(template=t, acceptance=dict(report_sha256='a'*64))
    monkeypatch.setattr(annotation_templates, 'match', lambda *args: (entry, constraints, audit))
    result = complete(receipt, 'lease', dict(source_sha256=digest(path), constraints=constraints),
                      'role-test', reference_template=t)
    assert not path.exists()
    with h5py.File(result['published'], 'r') as h:
        assert read_embedded(h)['constraints'] == constraints
        history = json.loads(h['trajectory_constraints/history'][()])
        assert history[-1]['programmatic_audit']['reference_roles'] == audit['reference_roles']
        assert json.loads(h['trajectory_constraints/acceptance'][()])['template_sha256'] == template_hash(t)
