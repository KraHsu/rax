"""Observed compliance evidence stays visible without weakening formal gates."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import discover_robotwin_observed_constraints as runner
from scripts import search_robotwin_constraint_counterexamples as formal
from services.robotwin import observed_discovery as observed
from services.robotwin.collection import atomic_json
from local_tests.unit.test_robotwin_counterexample_search import reference, native_result, early_close


def test_tracking_invalid_actual_compliant_failure_is_retained():
    ref = reference()
    command = early_close(ref)
    command['trajectory']['left']['pose'][1][0] += .02
    actual = copy.deepcopy(ref['trajectory'])
    raw, actual = native_result(ref, command['trajectory'], actual, success=False)
    result = observed.classify_observed(ref, command, raw, actual)
    assert result['observed_actual_false_accept'] and result['observationally_valid']
    assert not result['strict_eligible'] and not result['formal_evidence_eligible']
    assert result['strict'] == formal.classify_trial(ref, command, raw, actual)


def test_planner_failed_complete_actual_compliant_failure_is_not_discarded():
    ref = reference()
    command = early_close(ref)
    raw, actual = native_result(ref, command['trajectory'], bad_planner=True)
    result = observed.classify_observed(ref, command, raw, actual)
    assert result['observed_actual_false_accept'] and result['observed_command_failure']
    assert not result['strict']['planner_success'] and not result['strict_eligible']
    assert not result['control_execution_reliable']


def test_actual_rejected_native_success_is_valid_observational_control():
    ref = reference()
    command = dict(trajectory=ref['trajectory'], source='expert', family='reference_command')
    actual = copy.deepcopy(ref['trajectory'])
    actual['left']['pose'][2][0] += .0082
    raw, actual = native_result(ref, command['trajectory'], actual, success=True)
    result = observed.classify_observed(ref, command, raw, actual)
    assert not result['strict_eligible'] and result['observed_actual_success_rejected']
    assert result['control_execution_reliable'] and result['strict']['commanded_constraint_pass']
    assert result['strict']['tracking']['limits']['position_m'] == .005


def test_incomplete_execution_is_not_a_native_failure_or_success_label():
    ref = reference()
    command = early_close(ref)
    raw = dict(execution_complete=False, native_success=False)
    result = observed.classify_observed(ref, command, raw, {'left': {}, 'right': {}})
    assert not result['observationally_valid'] and not result['observed_actual_false_accept']
    assert result['strict']['native_success'] is None


@pytest.mark.parametrize('change', [dict(split='holdout'), dict(holdout_allowed=True),
                                   dict(publication_allowed=True), dict(automatic_retries=1),
                                   dict(seed=10001), dict(total_native_slots_max=26)])
def test_design_rejects_formal_relabeling_retries_and_other_scenes(tmp_path, change):
    design = dict(version=1, kind=observed.KIND, purpose=observed.PURPOSE, split='dev',
                  task='open_microwave', seed=10000, confidence_eligible=False, holdout_allowed=False,
                  publication_allowed=False, acceptance_ready=False, reference_recapture=False,
                  scene_replacement=False, adaptive_feedback=False, automatic_retries=0,
                  total_native_slots_max=25, generator_budget=20, unfiltered_signed_control_budget=4,
                  generator_seed=20261030, strict_tracking=observed.TRACKING)
    design.update(change)
    with pytest.raises(ValueError):
        observed.validate_design(tmp_path, design)


def test_formal_verifier_rejects_renamed_observed_manifest(tmp_path):
    atomic_json(tmp_path/'search_manifest.json', dict(version=1, task='open_microwave',
                kind=observed.KIND, purpose=observed.PURPOSE, formal_evidence_eligible=False))
    with pytest.raises(ValueError, match='protocol'):
        formal.verify_campaign(tmp_path)


def finite_manifest():
    return dict(trial_plan=[dict(trial_id='control_0', kind='control')]
                + [dict(trial_id=f'candidate_{i:04d}', kind='candidate') for i in range(24)])


@pytest.mark.parametrize('stop_index', [0, 3])
def test_native_attempt_failure_stops_once_without_retry_or_replacement(tmp_path, monkeypatch, stop_index):
    calls = []

    def execute(root, manifest, slot, factory, reference, control):
        calls.append(slot['trial_id'])
        good = len(calls)-1 != stop_index
        return {}, {}, dict(observationally_valid=good, control_execution_reliable=good)

    monkeypatch.setattr(runner, 'execute_slot', execute)
    assert runner.execute_plan(tmp_path, finite_manifest(), {}, object()) == 2
    assert len(calls) == stop_index+1 and len(set(calls)) == len(calls)
    assert json.loads((tmp_path/'state.json').read_text())['status'] == 'blocked'


def test_tracking_invalid_control_and_complete_candidate_failures_do_not_change_budget(tmp_path, monkeypatch):
    calls = []

    def execute(root, manifest, slot, factory, reference, control):
        calls.append(slot['trial_id'])
        return {}, {}, dict(observationally_valid=True, control_execution_reliable=slot['kind'] == 'control',
                            strict_eligible=False, observed_actual_false_accept=slot['kind'] == 'candidate')

    monkeypatch.setattr(runner, 'execute_slot', execute)
    assert runner.execute_plan(tmp_path, finite_manifest(), {}, object()) == 0
    assert len(calls) == len(set(calls)) == 25


def test_existing_output_cannot_resume_even_if_empty(tmp_path, monkeypatch):
    from services.robotwin import runtime
    from scripts import validate_robotwin_constraint_acceptance as driver
    design = tmp_path/'design.json'
    atomic_json(design, dict(native_commit='native'))
    output = tmp_path/'attempt'; output.mkdir()
    monkeypatch.setattr(runner, 'validate_design', lambda *a: ({}, [], {}))
    monkeypatch.setattr(runtime, 'source_commit', lambda: 'native')
    monkeypatch.setattr(driver, 'training_exclusions', lambda *a: {})
    monkeypatch.setattr(runtime, 'initialize_runtime', lambda: pytest.fail('Must not initialize GPU'))
    with pytest.raises(FileExistsError):
        runner.run(SimpleNamespace(design=design, output=output, training_root=tmp_path, execute=True))
    assert list(output.iterdir()) == []


@pytest.mark.parametrize('record_exists', [False, True])
def test_unindexed_partial_or_committed_directory_is_not_silently_ignored(tmp_path, monkeypatch, record_exists):
    design = dict(task='open_microwave', seed=10000, scene_sha256='scene', trial_plan=finite_manifest()['trial_plan'])
    atomic_json(tmp_path/'design_plan.json', design)
    manifest = dict(version=1, kind=observed.KIND, purpose=observed.PURPOSE, task='open_microwave',
                    scene_seed=10000, scene_sha256='scene', split='dev', trial_plan=design['trial_plan'],
                    formal_evidence_eligible=False, acceptance_ready=False, publication_allowed=False,
                    runtime_sha256=observed.runtime_fingerprint(),
                    design_sha256=observed.file_hash(tmp_path/'design_plan.json'))
    atomic_json(tmp_path/observed.MANIFEST, manifest)
    monkeypatch.setattr(observed, 'validate_design', lambda *a: ({}, [], {}))
    directory = tmp_path/'trials/control_0'; directory.mkdir(parents=True)
    if record_exists:
        atomic_json(directory/'record.json', {})
    with pytest.raises(ValueError, match='Uncommitted'):
        observed.verify_observed_campaign(tmp_path)
