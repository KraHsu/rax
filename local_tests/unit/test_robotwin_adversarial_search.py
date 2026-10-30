"""Inside-set proposals preserve frozen label activation and explicit provenance."""
import copy
import math

import numpy as np
import pytest

from services.robotwin.acceptance import json_hash, trajectory_violations
from services.robotwin.adversarial_search import (
    blend, generate_candidates, score_candidate, search_manifest,
)


def scene():
    n = 17
    trace = {}
    for arm in ('left', 'right'):
        pose = np.tile([0., 0., .8, 1., 0., 0., 0.], (n, 1))
        grip = np.ones(n)
        if arm == 'left':
            pose[:, 0] = np.linspace(0, .08, n)
            pose[:, 1] = .01*np.sin(np.linspace(0, math.pi, n))
            grip[6:12] = 0
        trace[arm] = dict(pose=pose.tolist(), gripper=grip.tolist())
    constraints = []
    def add(cid, kind, arm, params, activation, origin=(0., 0., 0.)):
        constraints.append(dict(id=cid, type=kind, subject=arm+'_eef',
            reference_frame=dict(origin_m=list(origin), quaternion_wxyz=[1., 0., 0., 0.]),
            parameters=params, activation=activation,
            penalty=dict(kind='squared_hinge', scale=.01, weight=1.),
            evidence=dict(source='known_geometry', status='accepted', report_id='synthetic', rationale='Test geometry only')))
    for arm in trace:
        add(arm+'_height', 'axis_position_interval', arm, dict(axis=[0., 0., 1.], bounds=[.79, .81]),
            dict(mode='interval', start=0, end=n))
    add('close', 'gripper_interval', 'left', dict(bounds=[0., .1]),
        dict(mode='reference_event', start=6, end=7, event=dict(frame=6, direction='decreasing', threshold=.5)))
    add('close_location', 'point_distance_interval', 'left', dict(bounds=[0., .005]),
        dict(mode='reference_event', start=6, end=7, event=dict(frame=6, direction='decreasing', threshold=.5)),
        trace['left']['pose'][6][:3])
    add('end', 'point_distance_interval', 'left', dict(bounds=[0., .005]),
        dict(mode='endpoint', start=n-1, end=n), trace['left']['pose'][-1][:3])
    bundle = dict(schema_version='1.3',
                  action_contract=dict(position='absolute', rotation='absolute', quaternion_order='wxyz', gripper='absolute_open_fraction', frame='world'),
                  provenance=dict(split='validation', variant='clean_heldout', recording_id='test', source='acceptance_reference_not_for_training'),
                  constraints=constraints)
    return trace, bundle


def test_scoring_uses_exact_reference_event_not_generated_close():
    reference, bundle = scene()
    candidate = copy.deepcopy(reference)
    candidate['left']['gripper'] = [1.]*17
    result = score_candidate(reference, bundle, candidate)
    assert not result['commanded_constraint_pass']
    assert result['max_violation'] == pytest.approx(.9)
    # Conversely, closing too early is admitted by this incomplete annotation.
    candidate['left']['gripper'][1:] = [0.]*16
    result = score_candidate(reference, bundle, candidate)
    assert result['commanded_constraint_pass']
    assert result['native_outcome'] == 'unknown_until_rollout'
    assert result['constraints'] == trajectory_violations(bundle, reference, candidate)['constraints']


def test_candidates_are_deterministic_nonexpert_and_leave_inputs_untouched():
    reference, bundle = scene()
    before = copy.deepcopy((reference, bundle))
    rows = generate_candidates(reference, bundle, seed=12, budget=24)
    again = generate_candidates(reference, bundle, seed=12, budget=24)
    assert rows == again
    assert (reference, bundle) == before
    assert len(rows) == 24
    assert len({r['candidate_id'] for r in rows}) == 24
    assert {'stall_then_jump', 'shortcut', 'phase_reorder', 'early_close', 'bimanual_differential'} <= {r['family'] for r in rows}
    assert all(r['commanded_constraint_pass'] for r in rows)
    assert any(r['score']['novelty'] > .5 for r in rows)
    for row in rows:
        for arm in reference:
            assert row['trajectory'][arm]['pose'][0] == reference[arm]['pose'][0]
            assert row['trajectory'][arm]['gripper'][0] == reference[arm]['gripper'][0]
        assert row['bundle_sha256'] == json_hash(bundle)
        assert row['trajectory_sha256'] == json_hash(row['trajectory'])
    receipt = search_manifest(reference, bundle, seed=12, budget=24, candidates=rows)
    assert receipt['bit_generator'] == 'PCG64'
    assert len(receipt['candidates']) == 24
    assert receipt['search_module_sha256']


def test_no_progress_lower_bound_is_exposed_without_retiming_reference():
    reference, bundle = scene()
    # A full directional cone admits zero motion; this is a search finding,
    # never a presumed task failure. Reference activation remains unchanged.
    bundle['constraints'] = [c for c in bundle['constraints'] if c['id'].endswith('height')]
    bundle['constraints'].append(dict(id='direction', type='window_direction_cone', subject='left_eef',
        reference_frame=dict(origin_m=[0., 0., .8], quaternion_wxyz=[1., 0., 0., 0.]),
        parameters=dict(axis=[1., 0., 0.], bounds=[0., .001], half_angle_rad=.3,
                        window_steps=4, minimum_window_steps=2, minimum_reference_progress_m=.005, fallback_radius_m=.2),
        activation=dict(mode='interval', start=1, end=17),
        penalty=dict(kind='squared_hinge', scale=.01, weight=1.),
        evidence=dict(source='known_geometry', status='accepted', report_id='test', rationale='Direction only')))
    candidate = copy.deepcopy(reference)
    candidate['left']['pose'] = [reference['left']['pose'][0] for _ in range(17)]
    score = score_candidate(reference, bundle, candidate)
    assert score['commanded_constraint_pass']
    assert next(c for c in score['constraints'] if c['id']=='direction')['samples'] == 16


def test_model_seed_preserves_provenance_and_marks_any_search_transform():
    reference, bundle = scene()
    model = copy.deepcopy(reference)
    model['left']['pose'][8][0] += .02
    provenance = dict(checkpoint_sha256='a'*64, source_run_sha256='b'*64,
                      source_kind='model_receding_horizon_snapshot', raw_chunk_sha256='c'*64,
                      suffix='frozen_reference_not_policy')
    external = [dict(candidate_id='inference-1', source='model', family='model_receding_horizon_snapshot',
                     trajectory=model, model_provenance=provenance)]
    rows = generate_candidates(reference, bundle, seed=3, budget=8, model_candidates=external)
    models = [r for r in rows if r['source']=='model']
    assert models
    assert models[0]['model_provenance']['source_kind'] == provenance['source_kind']
    assert models[0]['model_provenance']['suffix'] == provenance['suffix']
    assert models[0]['model_provenance']['unmodified_model_trajectory_sha256'] == json_hash(model)
    assert models[0]['model_provenance']['candidate_is_full_policy_rollout'] is False
    assert external[0]['model_provenance'] == provenance
    bad = copy.deepcopy(external);bad[0]['model_provenance'] = {'source_kind':'claimed_model'}
    with pytest.raises(ValueError, match='checkpoint/run'):
        generate_candidates(reference, bundle, seed=3, budget=8, model_candidates=bad)


@pytest.mark.parametrize('constrained_frame,expected_full', [(8, True), (6, False)])
def test_only_unblended_closed_loop_candidates_are_full_policy_rollouts(constrained_frame, expected_full):
    reference, bundle = scene()
    model = copy.deepcopy(reference)
    model['left']['pose'][constrained_frame][0] += .02
    provenance = dict(checkpoint_sha256='a'*64, source_run_sha256='b'*64,
                      source_kind='model_closed_loop', full_policy_rollout=True)
    external = [dict(candidate_id='closed-loop-1', source='model', family='model_closed_loop',
                     trajectory=model, model_provenance=provenance)]
    rows = generate_candidates(reference, bundle, seed=3, budget=8, model_candidates=external)
    chosen = next(row for row in rows if row['source']=='model')
    assert chosen['model_provenance']['candidate_is_full_policy_rollout'] is expected_full
    assert (chosen['parameters']['reference_blend'] == 1) is expected_full
    assert chosen['model_provenance']['full_policy_rollout'] is True
    assert external[0]['model_provenance'] == provenance


def test_feedback_only_accepts_native_verified_inside_set_failures():
    reference, bundle = scene()
    candidate = copy.deepcopy(reference);candidate['left']['gripper'][1:] = [0.]*16
    parent = dict(evidence_id='dev/trial-17', native_success=False, planner_success=True,
                  execution_complete=True, realized=True, trajectory=candidate)
    rows = generate_candidates(reference, bundle, seed=8, budget=32, feedback_parents=[parent])
    children = [r for r in rows if r['family']=='counterexample_neighborhood']
    assert children and all(r['parameters']['evidence_id']=='dev/trial-17' for r in children)
    assert all(r['score']['native_outcome']=='unknown_until_rollout' for r in children)
    parent['realized'] = False
    with pytest.raises(ValueError, match='realized native failure'):
        generate_candidates(reference, bundle, seed=8, feedback_parents=[parent])
    parent['realized'] = True;parent['trajectory']['left']['pose'][6][2] += 1
    with pytest.raises(ValueError, match='inside-set'):
        generate_candidates(reference, bundle, seed=8, feedback_parents=[parent])


def test_initial_state_alignment_and_quaternion_geodesic_blending():
    reference, bundle = scene()
    candidate = copy.deepcopy(reference)
    candidate['left']['pose'][0][0] += .001
    with pytest.raises(ValueError, match='initial'):
        score_candidate(reference, bundle, candidate)
    candidate = copy.deepcopy(reference);candidate['right']['gripper'].pop()
    with pytest.raises(ValueError):
        score_candidate(reference, bundle, candidate)
    candidate = copy.deepcopy(reference)
    candidate['left']['pose'][4][3:] = [0., 0., 0., -1.]
    halfway = blend(reference, candidate, .5)
    assert np.linalg.norm(halfway['left']['pose'][4][3:]) == pytest.approx(1.)
    assert abs(halfway['left']['pose'][4][3]) == pytest.approx(math.sqrt(.5))
    assert halfway['left']['pose'][0] == reference['left']['pose'][0]


def test_manifest_rejects_changed_candidate_and_context():
    reference, bundle = scene()
    rows = generate_candidates(reference, bundle, seed=2, budget=4)
    changed = copy.deepcopy(rows)
    changed[0]['trajectory']['left']['pose'][7][0] += .001
    with pytest.raises(ValueError, match='immutable search'):
        search_manifest(reference, bundle, seed=2, budget=4, candidates=changed)
    changed = copy.deepcopy(rows);changed[0]['parameters']['reference_blend'] = .0001
    with pytest.raises(ValueError, match='metadata changed'):
        search_manifest(reference, bundle, seed=2, budget=4, candidates=changed)
    with pytest.raises(ValueError, match='immutable search'):
        search_manifest(reference, bundle, seed=3, budget=4, candidates=rows)


def test_coupled_grasp_offsets_include_frozen_and_compensated_continuations():
    reference, bundle = scene()
    rows = generate_candidates(reference, bundle, seed=2, budget=32)
    pair = [r for r in rows if r['family'].startswith('grasp_offset_')]
    assert {'grasp_offset_frozen_continuation', 'grasp_offset_compensated_transport'} <= {r['family'] for r in pair}
    for row in pair:
        assert row['parameters']['close_frame'] == 6
        assert row['parameters']['release_frame'] == 12
        assert row['score']['native_outcome'] == 'unknown_until_rollout'
        assert row['commanded_constraint_pass']
