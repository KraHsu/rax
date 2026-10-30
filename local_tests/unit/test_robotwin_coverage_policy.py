"""Data-only coverage calibration preserves reference masks and loss scales."""
import copy
import hashlib
import json
import math

import h5py
import numpy as np
import pytest

from services.robotwin.acceptance import trajectory_violations
from services.robotwin.constraint_coverage import chunk_candidate, coverage_masks
from services.robotwin.full_trajectory import (
    POLICY, complete_constraints, resolve_coverage_policy, validate_coverage_policy,
)
from services.robotwin.programmatic_annotation import instantiate, validate_template
from test_robotwin_programmatic_annotation import episode, template


def fixture():
    n = 17
    left = np.tile([0., 0., .8, 1., 0., 0., 0.], (n, 1))
    left[1:9, 0] = np.arange(1, 9)*.01
    left[9:, 0] = .08
    left[9:13, 1] = np.arange(1, 5)*.01
    left[13:, 1] = .04
    return {'endpose/left_endpose': left,
            'endpose/right_endpose': np.tile([0., 0., .8, 1., 0., 0., 0.], (n, 1))}


def trajectory(handle):
    return {arm: dict(pose=handle[f'endpose/{arm}_endpose'].tolist(), gripper=[1.]*17)
            for arm in ('left', 'right')}


def score(constraints, reference, candidate):
    bundle = dict(schema_version='1.3', constraints=constraints,
                  action_contract=dict(position='absolute', rotation='absolute', quaternion_order='wxyz',
                                       gripper='absolute_open_fraction', frame='world'),
                  provenance=dict(split='validation', variant='clean_heldout', recording_id='test',
                                  source='acceptance_reference_not_for_training'))
    return trajectory_violations(bundle, reference, candidate)


def test_legacy_default_numeric_and_audit_regression():
    # Captured before introducing overrides: default labels and audit are exact.
    constraints, audit = complete_constraints(fixture(), [], 'coverage-policy-regression')
    serialized = json.dumps([constraints, audit], sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert hashlib.sha256(serialized.encode()).hexdigest() == '22160a60e21da871cc41ac76ce14efa2d6a3959ae10f65ff738da18beec88b35'
    assert complete_constraints(fixture(), [], 'coverage-policy-regression', policy=None) == (constraints, audit)
    defaults = {key: POLICY[key] for key in ('half_angle_rad', 'hold_margin_m', 'direction_slack_m')}
    explicit, explicit_audit = complete_constraints(fixture(), [], 'coverage-policy-regression', policy=defaults)
    assert explicit == constraints and explicit_audit['policy'] == audit['policy']
    audit['policy']['hold_margin_m'] = 999
    assert POLICY['hold_margin_m'] == .02  # Audit callers cannot mutate the global default.


def test_override_cannot_change_turn_segmentation_activation_or_chunk_coverage():
    handle = fixture()
    for i in range(9, 13):
        handle['endpose/left_endpose'][i, :2] = [.08+(i-8)*.005, (i-8)*math.sqrt(3)*.005]
    handle['endpose/left_endpose'][13:, :2] = handle['endpose/left_endpose'][12, :2]
    before = copy.deepcopy(handle)
    original, _ = complete_constraints(handle, [], 'fixed-partition')
    override = dict(half_angle_rad=math.radians(85), hold_margin_m=.08, direction_slack_m=.003)
    widened, audit = complete_constraints(handle, [], 'fixed-partition', policy=override)
    assert [(c['id'], c['activation']) for c in widened] == [(c['id'], c['activation']) for c in original]
    assert coverage_masks(widened, 17) == coverage_masks(original, 17)
    for start in range(16):
        for horizon in (1, 2, 4, 8, 16):
            selected = [c for c in widened if chunk_candidate(c, start, horizon)]
            masks = coverage_masks(selected, 17)
            for arm in masks:
                assert all(masks[arm]['position_or_motion'][start+1:min(start+horizon+1, 17)])
    assert audit['complete'] and audit['coverage_override'] == override
    assert audit['reference_partition_policy']['half_angle_rad'] == POLICY['half_angle_rad']
    assert audit['policy']['half_angle_rad'] == override['half_angle_rad']
    for key in handle:
        np.testing.assert_array_equal(handle[key], before[key])
    assert override == dict(half_angle_rad=math.radians(85), hold_margin_m=.08, direction_slack_m=.003)


def test_hold_fallback_and_slack_enlarge_allowed_set_without_rescaling_loss():
    handle = fixture()
    original, _ = complete_constraints(handle, [], 'monotonic')
    widened, audit = complete_constraints(handle, [], 'monotonic', policy=dict(hold_margin_m=.05, direction_slack_m=.002))
    assert [c['penalty'] for c in original] == [c['penalty'] for c in widened]
    assert audit['penalty_scales_m'] == dict(stationary=.02, direction=.005)
    reference = trajectory(handle)
    allowed = copy.deepcopy(reference)
    allowed['right']['pose'] = [[p[0]+(.03 if i else 0), *p[1:]] for i, p in enumerate(allowed['right']['pose'])]
    allowed['left']['pose'][1][1] += .075  # The fixed short-window fallback, not a direction branch.
    assert not score(original, reference, allowed)['passed']
    assert score(widened, reference, allowed)['passed']
    rng = np.random.default_rng(74)
    for _ in range(30):
        candidate = copy.deepcopy(reference)
        for arm in candidate:
            poses = np.asarray(candidate[arm]['pose'])
            poses[1:, :3] += rng.normal(0, .025, (16, 3))
            candidate[arm]['pose'] = poses.tolist()
        old = score(original, reference, candidate)['constraints']
        new = score(widened, reference, candidate)['constraints']
        assert all(b['max_violation'] <= a['max_violation']+1e-12 for a, b in zip(old, new))
    # Equal 5 mm exceedance beyond two different hold regions has equal loss.
    old_hold = next(c for c in original if c['subject'] == 'right_eef')
    new_hold = next(c for c in widened if c['subject'] == 'right_eef')
    losses = []
    for constraint, complete in ((old_hold, original), (new_hold, widened)):
        candidate = copy.deepcopy(reference)
        for pose in candidate['right']['pose'][1:]:
            pose[0] = constraint['parameters']['bounds'][1]+.005
        residual = next(c['max_violation'] for c in score(complete, reference, candidate)['constraints']
                        if c['id'] == constraint['id'])
        losses.append((residual/constraint['penalty']['scale'])**2)
    assert losses[0] == pytest.approx(losses[1])


def test_angle_expands_forward_cone_but_retains_fixed_fallback_switches():
    handle = fixture()
    original, _ = complete_constraints(handle, [], 'angle')
    widened, _ = complete_constraints(handle, [], 'angle', policy=dict(half_angle_rad=math.radians(75)))
    reference = trajectory(handle)
    candidate = copy.deepcopy(reference)
    for i in range(1, 9):
        candidate['left']['pose'][i][:2] = [.01*i*math.cos(math.radians(70)), .01*i*math.sin(math.radians(70))]
    assert score(original, reference, candidate)['constraints'][0]['max_violation'] > 1e-4
    assert score(widened, reference, candidate)['constraints'][0]['max_violation'] == 0
    assert original[0]['parameters']['minimum_window_steps'] == widened[0]['parameters']['minimum_window_steps']
    assert original[0]['penalty'] == widened[0]['penalty']


@pytest.mark.parametrize('bad', [None, {}, [], False, {'disable': True}, {'window_steps': 1},
    {'stationary_step_m': .01}, {'maximum_predicted_anchor_span': 1000}, {'scale': 100},
    {'hold_margin_m': True}, {'hold_margin_m': float('nan')}, {'hold_margin_m': float('inf')},
    {'hold_margin_m': .0009}, {'hold_margin_m': .10001}, {'direction_slack_m': -.001},
    {'direction_slack_m': .05001}, {'half_angle_rad': math.radians(4.9)}, {'half_angle_rad': math.pi/2}])
def test_unsafe_or_mask_changing_overrides_are_rejected(bad):
    with pytest.raises(ValueError, match='coverage_policy'):
        validate_coverage_policy(bad)
    t = template(); t['coverage_policy'] = bad
    with pytest.raises(ValueError, match='coverage_policy'):
        validate_template(t)


def test_template_instantiation_records_effective_policy_and_preserves_task_rules(tmp_path):
    t = template()
    changed = copy.deepcopy(t)
    changed['coverage_policy'] = dict(hold_margin_m=.04, direction_slack_m=.001)
    with h5py.File(episode(tmp_path/'episode.hdf5')) as handle:
        original, before = instantiate(handle, t)
        revised, audit = instantiate(handle, changed)
    assert audit['coverage']['complete'] and before['coverage']['complete']
    assert audit['coverage']['policy'] == resolve_coverage_policy(changed['coverage_policy'])
    assert audit['coverage']['reference_partition_policy']['stationary_step_m'] == POLICY['stationary_step_m']
    for old, new in zip(original, revised):
        if not old['id'].startswith('__full_motion_'):
            assert {k: v for k, v in old.items() if k != 'evidence'} == {k: v for k, v in new.items() if k != 'evidence'}
        assert old['penalty'] == new['penalty']
    assert 'coverage_policy' not in t
