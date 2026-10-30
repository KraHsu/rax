"""Acceptance playback uses actual native verdicts and fixed reference indices."""
from types import SimpleNamespace

import numpy as np
import pytest

from services.robotwin.acceptance import annotation_policy_fingerprint, evaluate_trial, validate_manifest
from services.robotwin.acceptance_replay import (
    ALIGNMENT, SUCCESS_RULE, arrays, command_rows, default_cases, intervene, replay,
    quaternion_distance, quaternion_multiply, tracking_report,
)
from services.robotwin.collection import atomic_json, scene_fingerprint
from services.robotwin.programmatic_annotation import template_hash
from scripts.validate_robotwin_constraint_acceptance import (
    bind, json_digest, planned_trials, rebuild_index, reference_bundle, trial_record,
)


def reference(length=10):
    pose = np.tile([0., 0., .8, 1., 0., 0., 0.], (length, 1))
    moving = pose.copy(); moving[:, 0] = np.linspace(0, .1, length)
    grip = np.ones(length); grip[3:] = 0
    return dict(left=dict(pose=moving.tolist(), gripper=grip.tolist()),
                right=dict(pose=pose.tolist(), gripper=np.ones(length).tolist()))


def template():
    return dict(version=1, task='adjust_bottle', rationale='Synthetic accepted-schema fixture',
                signatures=[['close']], rules=[
                    dict(id='terminal', type='point_distance_interval', subjects='active_arms',
                         selector=dict(kind='terminal'), origin='selection_end', orientation='world',
                         parameters={}, rationale='Terminal reference sphere'),
                    dict(id='grip', type='gripper_interval', subjects='active_arms',
                         selector=dict(kind='gripper_plateaus', state='closed', after='episode_start'),
                         origin='world', orientation='world', parameters={}, rationale='Nominal closed plateau'),
                ])


class FakeRobot:
    def __init__(self):
        self.left = [0., 0., .8, 1., 0., 0., 0., 1.]
        self.right = self.left.copy()

    def get_left_ee_pose(self): return self.left[:7]
    def get_right_ee_pose(self): return self.right[:7]
    def get_left_gripper_val(self): return self.left[7]
    def get_right_gripper_val(self): return self.right[7]
    def left_plan_path(self, target): return {'status': 'Success'}
    def right_plan_path(self, target): return {'status': 'Success'}


class FakeNative:
    def __init__(self, *, fail_at=None, planner_failure=False):
        self.robot = FakeRobot()
        self.scene = SimpleNamespace(get_all_actors=lambda: [], get_all_articulations=lambda: [])
        self.take_action_cnt = 0; self.eval_success = False; self.step_lim = 1
        self.fail_at = fail_at
        if planner_failure:
            self.robot.left_plan_path = lambda target: {'status': 'Failure'}

    def check_success(self):
        return self.robot.left[0] > .035

    def take_action(self, action, action_type):
        assert action_type == 'ee'
        if self.eval_success or self.take_action_cnt == self.step_lim: return
        self.take_action_cnt += 1
        if self.take_action_cnt == self.fail_at: raise RuntimeError('Synthetic execution error')
        self.robot.left_plan_path(action[:7]); self.robot.right_plan_path(action[8:15])
        self.robot.left = action[:8].tolist(); self.robot.right = action[8:].tolist()
        if self.check_success(): self.eval_success = True

    def close_env(self): pass


def test_case_design_is_fixed_and_contains_all_families_without_outcome_labels():
    cases = default_cases()
    assert len(cases) == 18 and len({c['id'] for c in cases}) == 18
    assert {c['variant_family'] for c in cases} == {'position', 'orientation', 'gripper_timing', 'coupled'}
    assert all('native_success' not in c and 'expected_success' not in c for c in cases)
    assert len(planned_trials([10000, 10001, 10002], 2, cases)) == 60


def test_quaternion_distance_is_stable_near_zero_sign_invariant_and_correct_at_pi():
    identity = np.array([1., 0., 0., 0.])
    small = np.array([np.cos(5e-15), np.sin(5e-15), 0., 0.])
    assert quaternion_distance(small, identity) == pytest.approx(1e-14, abs=1e-17)
    nontrivial = np.array([.5, .5, .5, .5])
    shifted = quaternion_multiply(np.array([np.cos(5e-13), 0., np.sin(5e-13), 0.]), nontrivial)
    assert quaternion_distance(shifted, nontrivial) == pytest.approx(1e-12, abs=1e-15)
    assert quaternion_distance(nontrivial, -nontrivial) < 1e-15
    rotations = np.array([identity, small, [0., 0., 0., 1.]])
    np.testing.assert_allclose(quaternion_distance(-rotations, identity), [0., 1e-14, np.pi], rtol=0, atol=1e-15)


def test_every_intervention_preserves_initial_state_and_reference_length():
    ref = reference()
    for case in default_cases():
        modified = arrays(intervene(ref, case))
        assert command_rows(modified).shape == (10, 16)
        for arm in ('left', 'right'):
            np.testing.assert_array_equal(modified[arm]['pose'][0], ref[arm]['pose'][0])
            assert modified[arm]['gripper'][0] == ref[arm]['gripper'][0]
        np.testing.assert_array_equal(modified['right']['pose'], ref['right']['pose'])
        np.testing.assert_array_equal(modified['right']['gripper'], ref['right']['gripper'])


def test_actual_replay_keeps_all_frames_after_early_native_success_and_restores_predicate():
    native = FakeNative()
    result = replay(native, reference())
    assert result['execution_complete'] and result['planner_success']
    assert result['native_success'] and result['ever_success'] and result['final_success']
    assert len(result['actual']['left']['pose']) == 10 and native.take_action_cnt == 9
    assert result['alignment'] == ALIGNMENT
    assert native.check_success() is True


def test_planner_failure_is_not_a_valid_negative_and_execution_errors_stay_incomplete():
    result = replay(FakeNative(planner_failure=True), reference())
    assert result['execution_complete'] and not result['planner_success']
    result = replay(FakeNative(fail_at=3), reference())
    assert not result['execution_complete'] and result['error']
    assert len(result['actual']['left']['pose']) == 3


def test_ignored_small_intervention_cannot_pass_absolute_tracking_ceiling():
    ref = reference()
    case = dict(id='small', variant_family='position', position_axis=2, position_offset_m=.003)
    commanded = intervene(ref, case)
    assert tracking_report(ref, commanded)['realized']  # within 5 mm absolute ceiling
    assert not tracking_report(ref, commanded, baseline=ref, reference=ref)['realized']
    assert tracking_report(commanded, commanded, baseline=ref, reference=ref)['realized']


def test_ignored_small_rotation_and_gripper_changes_fail_relative_tracking():
    ref = reference()
    rotated = intervene(ref, dict(variant_family='orientation', rotation_axis=0, rotation_rad=np.deg2rad(2)))
    assert tracking_report(ref, rotated)['realized']
    assert not tracking_report(ref, rotated, baseline=ref, reference=ref)['realized']
    assert tracking_report(rotated, rotated, baseline=ref, reference=ref)['realized']
    gripper = reference()
    gripper['left']['gripper'][3:] = [.03]*7
    assert tracking_report(ref, gripper)['realized']
    assert not tracking_report(ref, gripper, baseline=ref, reference=ref)['realized']
    assert tracking_report(gripper, gripper, baseline=ref, reference=ref)['realized']


def test_relative_rotations_use_world_composition_and_ignore_quaternion_sign():
    ref = reference()
    ref['left']['pose'] = np.asarray(ref['left']['pose'])
    ref['left']['pose'][:, 3:] = [np.cos(.3), 0, np.sin(.3), 0]
    ref['left']['pose'] = ref['left']['pose'].tolist()
    commanded = intervene(ref, dict(variant_family='orientation', rotation_axis=0, rotation_rad=.02))
    actual = arrays(commanded)
    actual['left']['pose'][:, 3:] *= -1
    assert tracking_report(actual, commanded, baseline=ref, reference=ref)['realized']
    short = reference(9)
    with pytest.raises(ValueError, match='complete command-index alignment'):
        tracking_report(actual, commanded, baseline=short, reference=ref)


def test_reference_artifact_is_validation_only_and_protected_from_training_reader():
    from services.robotwin.constraint_schema import validate_bundle
    bundle, _ = reference_bundle(template(), reference(), 10000, 'testscene')
    assert bundle['provenance']['split'] == 'validation'
    assert bundle['provenance']['variant'] == 'clean_heldout'
    with pytest.raises(ValueError, match='train / clean_50'):
        validate_bundle(bundle)


def test_real_record_roundtrip_is_recomputed_by_cpu_gate_and_artifacts_are_hashed(tmp_path):
    task_template = template(); ref = reference()
    fingerprint = scene_fingerprint(dict(actors=[], task_articulations=[]))
    bundle, _ = reference_bundle(task_template, ref, 10000, fingerprint)
    reference_path = tmp_path/'references/seed10000.json'
    atomic_json(reference_path, dict(bundle=bundle, trajectory=ref,
                                    provenance=dict(task='adjust_bottle', seed=10000, scene_sha256=fingerprint,
                                                    mode='demo_clean', split='validation', source_episode_sha256=json_digest(ref))))
    planned = planned_trials([10000], 2, [])[0]
    manifest = dict(version=1, task='adjust_bottle', template=task_template, template_sha256=template_hash(task_template),
                    annotation_policy_sha256=annotation_policy_fingerprint(), native_commit='a'*40,
                    mode='demo_clean', split='validation', alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE,
                    excluded_training_seeds=[0], excluded_training_scene_sha256=['training_scene'], trial_plan=[planned])
    validate_manifest(manifest)
    record, raw, actual = trial_record(tmp_path, manifest, planned, reference_path, lambda seed: FakeNative(), None)
    assert raw['native_success'] and raw['realized']
    checked = evaluate_trial(record, tmp_path, manifest)
    assert checked['eligible'] and checked['constraint_pass']
    assert (tmp_path/'trials.jsonl').exists()
    with (tmp_path/record['outcome_path']).open('a') as handle: handle.write(' ')
    with pytest.raises(ValueError, match='Changed outcome'):
        evaluate_trial(record, tmp_path, manifest)


def test_resume_configuration_cannot_change_after_results(tmp_path):
    path = tmp_path/'configuration.json'
    bind(path, dict(seeds=3)); bind(path, dict(seeds=3))
    with pytest.raises(ValueError, match='Frozen'):
        bind(path, dict(seeds=4))


def test_resume_index_uses_only_committed_records_not_partial_trials(tmp_path):
    atomic_json(tmp_path/'trials/complete/record.json', dict(trial_id='complete'))
    atomic_json(tmp_path/'trials/partial/outcome.json', dict(native_success=True))
    (tmp_path/'trials.jsonl').write_text('interrupted index')
    rebuild_index(tmp_path)
    assert (tmp_path/'trials.jsonl').read_text().strip() == '{"trial_id": "complete"}'
