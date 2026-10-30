"""Negative evidence is measured, complete, held out, and version-bound."""
import copy
import json

import numpy as np
import pytest

from services.robotwin import acceptance as gate
from services.robotwin.constraint_schema import HDF5_KEY, SCHEMA_VERSION
from services.robotwin.programmatic_annotation import instantiate, template_hash
from services.robotwin.acceptance_replay import intervene, tracking_report


def replay_adapter():
    arms = {}
    for arm, prefix in [('left', 'fl'), ('right', 'fr')]:
        arms[arm] = dict(physics_urdf_path='/synthetic/robot.urdf', physics_urdf_sha256='a'*64,
                         planner_urdf_path='/synthetic/robot.urdf', planner_urdf_sha256='a'*64,
                         planner_yaml=f'/synthetic/{arm}.yml', planner_yaml_sha256='b'*64,
                         root_link='footprint', arm_base=prefix+'_base_link', ee_link=prefix+'_link6',
                         ee_joint=prefix+'_joint6', fixed_joint=prefix+'_base_joint',
                         xyz=[0., 0., 0.], rpy=[0., 0., 0.], frame_bias=[0., 0., 0.],
                         rotation=np.eye(3).tolist(), quaternion_order='wxyz')
    return dict(version=gate.REPLAY_ADAPTER_VERSION, arms=arms)


def template():
    def rule(name, kind, selector, parameters, origin='world'):
        return dict(id=name, type=kind, subjects='active_arms', selector=selector,
                    origin=origin, orientation='world', parameters=parameters, rationale='Synthetic gate fixture')
    return dict(version=1, task='adjust_bottle', rationale='Synthetic only', signatures=[['close']], rules=[
        rule('close', 'point_distance_interval', dict(kind='events', direction='decreasing', occurrence='first'), {}, 'selection_start'),
        rule('grip', 'gripper_interval', dict(kind='gripper_plateaus', state='closed', after='first_close'), {}),
        rule('height', 'axis_position_interval', dict(kind='terminal'), dict(axis=[0, 0, 1])),
        rule('angle', 'axis_angle_interval', dict(kind='terminal'), dict(local_axis=[0, 0, 1], reference_axis=[0, 0, 1])),
    ])


def reference(seed, t):
    trajectory = {}
    for arm in ('left', 'right'):
        pose = np.tile([0., 0., .8, 1., 0., 0., 0.], (5, 1))
        grip = np.ones(5)
        if arm == 'left':
            pose[:, 2] += np.arange(5)*.01
            grip[2:] = 0
        trajectory[arm] = dict(pose=pose.tolist(), gripper=grip.tolist())
    b = dict(schema_version=SCHEMA_VERSION,
             action_contract=dict(position='absolute', rotation='absolute', quaternion_order='wxyz',
                                  gripper='absolute_open_fraction', frame='world'),
             provenance=dict(split='train', variant='clean_50', recording_id=f'heldout{seed}',
                             source='acceptance_reference_not_for_training', native_success=True), constraints=[])
    provenance = dict(task=t['task'], seed=seed, scene_sha256=f'heldout-scene-{seed}',
                      initial_scene_sha256=f'heldout-scene-{seed}', mode='demo_clean', split='validation')
    class Handle(dict):
        pass
    h = Handle({HDF5_KEY: np.asarray(json.dumps(b))})
    h.attrs = dict(robotwin_expert_provenance=json.dumps(provenance))
    for arm, values in trajectory.items():
        h[f'endpose/{arm}_endpose'] = np.asarray(values['pose'])
        h[f'endpose/{arm}_gripper'] = np.asarray(values['gripper'])
    b['constraints'], _ = instantiate(h, t)
    b['provenance'].update(split='validation', variant='clean_heldout')
    return dict(bundle=b, trajectory=trajectory, provenance=provenance,
                replay_adapter=replay_adapter(), replay_adapter_applied=False)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))


def rewrite_records(root, records):
    (root/'trials.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))


def artifact(root, name, value):
    path = root/name
    write(path, value)
    return dict(path=name, sha256=gate.file_hash(path))


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path/'evidence'
    root.mkdir()
    t = template()
    manifest = dict(version=1, task=t['task'], template=t, template_sha256=template_hash(t),
                    annotation_policy_sha256=gate.annotation_policy_fingerprint(), native_commit='native-test-commit',
                    replay_adapter=replay_adapter(),
                    mode='demo_clean', split='validation', alignment=gate.ALIGNMENT, native_success_rule=gate.SUCCESS_RULE,
                    excluded_training_seeds=[0, 1, 2], excluded_training_scene_sha256=['train-a', 'train-b', 'train-c'],
                    trial_plan=[])
    records = []
    for seed in (100, 101, 102):
        ref = reference(seed, t)
        reference_artifact = artifact(root, f'references/{seed}.json', ref)
        for index, family in enumerate(('baseline', 'baseline', *gate.FAMILIES)):
            trial_id = f'{seed}-{index}'
            case = dict(id=trial_id, variant_family=family)
            success = family not in ('orientation', 'gripper_timing')
            if family == 'orientation':
                case.update(rotation_axis=0, rotation_rad=np.pi)
            elif family == 'gripper_timing':
                case.update(gripper_force=1.)
            elif family != 'baseline':
                case.update(position_axis=0, position_offset_m=.001)
                if family == 'coupled':
                    case.update(rotation_axis=0, rotation_rad=.01)
            manifest['trial_plan'].append(dict(trial_id=trial_id, seed=seed, variant_family=family, case=case))
            actual = intervene(ref['trajectory'], case)
            record = {k: manifest[k] for k in ('version', 'task', 'template_sha256', 'annotation_policy_sha256',
                                              'native_commit', 'mode', 'split', 'alignment', 'native_success_rule')}
            record.update(trial_id=trial_id, seed=seed, scene_sha256=ref['provenance']['scene_sha256'],
                          variant_family=family, bundle_sha256=gate.json_hash(ref['bundle']))
            outcome = {k: record[k] for k in ('trial_id', 'task', 'seed', 'scene_sha256', 'variant_family')}
            outcome.update(native_success=success, ever_success=success, final_success=success,
                           planner_success=True, realized=True, execution_complete=True,
                           replay_adapter=replay_adapter(), replay_adapter_applied=True,
                           planner_results=[dict(arm=arm, frame=frame, status='Success')
                                            for frame in range(1, 5) for arm in ('left', 'right')],
                           tracking=tracking_report(actual, actual, baseline=ref['trajectory'] if index else None,
                                                    reference=ref['trajectory'] if index else None))
            for name, item in [('reference', reference_artifact),
                               ('trajectory', artifact(root, f'trajectories/{trial_id}.json', dict(actual=actual, commanded=actual))),
                               ('outcome', artifact(root, f'outcomes/{trial_id}.json', outcome))]:
                record[name+'_path'] = item['path']
                record[name+'_sha256'] = item['sha256']
            records.append(record)
    write(root/'experiment_manifest.json', manifest)
    rewrite_records(root, records)
    return root, manifest, records


def test_balanced_realized_native_evidence_passes_pilot(evidence):
    root, _, _ = evidence
    report = gate.assess_evidence(root)
    assert report['ready']
    assert report['counts']['baselines'] == report['counts']['successful_variations'] == report['counts']['native_task_failures'] == 6
    assert report['false_accept_rate'] == report['false_reject_rate'] == 0


def test_native_failed_but_constraint_satisfying_trajectory_blocks(evidence):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    raw = json.loads((root/trial['outcome_path']).read_text())
    raw.update(native_success=False, ever_success=False, final_success=False)
    write(root/trial['outcome_path'], raw)
    trial['outcome_sha256'] = gate.file_hash(root/trial['outcome_path'])
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready'] and report['counts']['false_accepts'] == 1
    assert 'failed_variations_pass_constraints' in report['reasons']


def test_all_rejected_successes_cannot_pass(evidence):
    root, _, records = evidence
    for r in records:
        if r['variant_family'] != 'baseline':
            raw = json.loads((root/r['outcome_path']).read_text())
            raw.update(native_success=(r['variant_family'] in ('orientation', 'gripper_timing')),
                       ever_success=(r['variant_family'] in ('orientation', 'gripper_timing')))
            write(root/r['outcome_path'], raw)
            r['outcome_sha256'] = gate.file_hash(root/r['outcome_path'])
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready'] and report['false_reject_rate'] == 1


def test_planner_failure_is_excluded_not_a_negative_example(evidence):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'orientation')
    payload = json.loads((root/trial['outcome_path']).read_text())
    payload['planner_success'] = False
    payload['planner_results'][0]['status'] = 'Fail'
    write(root/trial['outcome_path'], payload)
    trial['outcome_sha256'] = gate.file_hash(root/trial['outcome_path'])
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready'] and report['counts']['native_task_failures'] == 5
    assert report['counts']['excluded_trials'] == 1


@pytest.mark.parametrize('mutation', ['missing', 'training_seed', 'policy', 'duplicate', 'random'])
def test_incomplete_or_contaminated_evidence_cannot_activate(evidence, mutation):
    root, manifest, records = evidence
    if mutation == 'missing':
        rewrite_records(root, records[:-1])
        assert not gate.assess_evidence(root)['ready']
        return
    if mutation == 'training_seed':
        manifest['excluded_training_seeds'].append(100)
    elif mutation == 'policy':
        manifest['annotation_policy_sha256'] = 'other-policy'
    elif mutation == 'random':
        manifest['mode'] = 'demo_randomized'
    elif mutation == 'duplicate':
        rewrite_records(root, records+[records[0]])
    write(root/'experiment_manifest.json', manifest)
    with pytest.raises(ValueError):
        gate.assess_evidence(root)


def test_tampered_artifact_blocks_even_when_record_claims_success(evidence):
    root, _, records = evidence
    write(root/records[0]['trajectory_path'], {'actual': {}})
    report = gate.assess_evidence(root)
    # The first baseline is also the tracking reference for the other five
    # trials of this seed; corruption invalidates all six, never hides a case.
    assert not report['ready'] and report['counts']['invalid_evidence'] == 6


def test_no_predicted_event_can_disable_reference_activation():
    ref = reference(100, template())
    measured = copy.deepcopy(ref['trajectory'])
    measured['left']['gripper'] = [1.]*5
    result = gate.trajectory_violations(ref['bundle'], ref['trajectory'], measured)
    assert not result['passed']
    assert any(r['type'] == 'gripper_interval' and r['max_violation'] > .9 for r in result['constraints'])


def test_reference_window_branch_is_not_selected_by_predicted_pause():
    ref = reference(100, template())
    measured = copy.deepcopy(ref['trajectory'])
    measured['left']['pose'][2][0] = 1
    # Source event position is fixed even though generated gripper never closes.
    measured['left']['gripper'] = [1.]*5
    result = gate.trajectory_violations(ref['bundle'], ref['trajectory'], measured)
    assert not result['passed']
    assert any(r['type'] == 'point_distance_interval' and r['max_violation'] > .9 for r in result['constraints'])


def test_missing_measured_frames_are_not_silently_clipped():
    ref = reference(100, template())
    measured = copy.deepcopy(ref['trajectory'])
    for arm in measured.values():
        arm['pose'].pop(); arm['gripper'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        gate.trajectory_violations(ref['bundle'], ref['trajectory'], measured)


def test_looser_bundle_cannot_hide_behind_original_template_hash(evidence):
    root, _, records = evidence
    r = records[0]
    ref = json.loads((root/r['reference_path']).read_text())
    ref['bundle']['constraints'][0]['parameters']['bounds'][1] = 100
    write(root/r['reference_path'], ref)
    for item in records:
        if item['seed'] == r['seed']:
            item['reference_sha256'] = gate.file_hash(root/item['reference_path'])
            item['bundle_sha256'] = gate.json_hash(ref['bundle'])
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready'] and any('instantiation' in e['error'] for e in report['errors'])


def test_realized_flag_cannot_hide_an_ignored_small_intervention(evidence):
    root, manifest, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    planned = next(p for p in manifest['trial_plan'] if p['trial_id'] == trial['trial_id'])
    planned['case']['position_offset_m'] = .003
    ref = json.loads((root/trial['reference_path']).read_text())
    # 3 mm is below the 5 mm absolute ceiling, but a completely ignored
    # intervention must fail relative tracking against the baseline.
    payload = dict(actual=ref['trajectory'], commanded=intervene(ref['trajectory'], planned['case']))
    write(root/trial['trajectory_path'], payload)
    trial['trajectory_sha256'] = gate.file_hash(root/trial['trajectory_path'])
    write(root/'experiment_manifest.json', manifest)
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready']
    assert any('Tracking/realization' in e['error'] for e in report['errors'])


def test_command_trace_must_match_frozen_case(evidence):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    payload = json.loads((root/trial['trajectory_path']).read_text())
    payload['commanded']['left']['pose'][-1][0] += .02
    write(root/trial['trajectory_path'], payload)
    trial['trajectory_sha256'] = gate.file_hash(root/trial['trajectory_path'])
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready']
    assert any('Commanded trace' in e['error'] for e in report['errors'])


def test_derived_comparison_is_numerically_tight_and_structurally_strict():
    same = gate.derived_values_equal
    assert same({'x': [1.0, 0.0], 'ok': True}, {'x': [1.0+5e-11, 5e-13], 'ok': True})
    assert not same(0.0, 1e-8)
    assert not same(1.0, 1.0+1e-8)
    assert not same(True, 1)
    assert not same(1, 1.0)
    assert not same(10**12, 10**12+1)
    assert not same([1.0], [1.0, 2.0])
    assert not same({'x': 1.0}, {'y': 1.0})
    assert not same(float('nan'), float('nan'))
    assert not same(float('inf'), float('inf'))


def test_tiny_cross_platform_derived_roundoff_is_accepted(evidence):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    payload = json.loads((root/trial['trajectory_path']).read_text())
    payload['commanded']['left']['pose'][-1][0] += 5e-13
    write(root/trial['trajectory_path'], payload)
    trial['trajectory_sha256'] = gate.file_hash(root/trial['trajectory_path'])
    raw = json.loads((root/trial['outcome_path']).read_text())
    raw['tracking']['absolute']['left']['position_m'] += 5e-13
    write(root/trial['outcome_path'], raw)
    trial['outcome_sha256'] = gate.file_hash(root/trial['outcome_path'])
    ref = json.loads((root/trial['reference_path']).read_text())
    ref['bundle']['constraints'][0]['parameters']['bounds'][1] += 5e-13
    write(root/trial['reference_path'], ref)
    for item in records:
        if item['seed'] == trial['seed']:
            item['reference_sha256'] = gate.file_hash(root/item['reference_path'])
            item['bundle_sha256'] = gate.json_hash(ref['bundle'])
    rewrite_records(root, records)
    assert gate.assess_evidence(root)['ready']


@pytest.mark.parametrize('value', [False, 1])
def test_tracking_decisions_remain_strict_despite_numerical_tolerance(evidence, value):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    raw = json.loads((root/trial['outcome_path']).read_text())
    raw['tracking']['realized'] = value
    write(root/trial['outcome_path'], raw)
    trial['outcome_sha256'] = gate.file_hash(root/trial['outcome_path'])
    rewrite_records(root, records)
    result = gate.assess_evidence(root)
    assert not result['ready']
    assert any('Tracking/realization' in e['error'] for e in result['errors'])


def test_family_name_alone_does_not_count_as_a_coupled_intervention(evidence):
    root, manifest, _ = evidence
    case = next(p['case'] for p in manifest['trial_plan'] if p['variant_family'] == 'coupled')
    del case['rotation_axis']
    del case['rotation_rad']
    write(root/'experiment_manifest.json', manifest)
    with pytest.raises(ValueError, match='family'):
        gate.assess_evidence(root)


@pytest.mark.parametrize('mutation', ['missing', 'different_urdf', 'invalid_rotation', 'xyzw'])
def test_adapter_manifest_identity_is_required_and_structurally_checked(evidence, mutation):
    root, manifest, _ = evidence
    arm = manifest['replay_adapter']['arms']['left']
    if mutation == 'missing':
        del manifest['replay_adapter']
    elif mutation == 'different_urdf':
        arm['planner_urdf_sha256'] = 'c'*64
    elif mutation == 'invalid_rotation':
        arm['rotation'][0][0] = -1.
    else:
        arm['quaternion_order'] = 'xyzw'
    write(root/'experiment_manifest.json', manifest)
    with pytest.raises(ValueError, match='adapter'):
        gate.assess_evidence(root)


@pytest.mark.parametrize('artifact_name', ['reference', 'outcome'])
def test_reference_and_outcome_adapter_source_hashes_must_match_manifest(evidence, artifact_name):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    path = root/trial[artifact_name+'_path']
    value = json.loads(path.read_text())
    value['replay_adapter']['arms']['left']['planner_yaml_sha256'] = 'c'*64
    write(path, value)
    for r in records:
        if r[artifact_name+'_path'] == trial[artifact_name+'_path']:
            r[artifact_name+'_sha256'] = gate.file_hash(path)
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready']
    assert any('Replay adapter' in e['error'] for e in report['errors'])


@pytest.mark.parametrize('artifact_name,value', [('reference', True), ('outcome', False)])
def test_adapter_only_applied_to_trials_not_reference_expert(evidence, artifact_name, value):
    root, _, records = evidence
    trial = next(r for r in records if r['variant_family'] == 'position')
    path = root/trial[artifact_name+'_path']
    payload = json.loads(path.read_text());payload['replay_adapter_applied'] = value
    write(path, payload)
    for r in records:
        if r[artifact_name+'_path'] == trial[artifact_name+'_path']:
            r[artifact_name+'_sha256'] = gate.file_hash(path)
    rewrite_records(root, records)
    report = gate.assess_evidence(root)
    assert not report['ready']
    assert any('Replay adapter' in e['error'] for e in report['errors'])
