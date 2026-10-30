"""CPU contract tests for the finite fresh-metadata smoke preset."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from scripts import validate_robotwin_constraint_acceptance as driver
from test_robotwin_acceptance import reference, template
from test_robotwin_success_metadata import environment, initialize


def args(tmp_path, **changes):
    value = dict(template=tmp_path/'template.json', output=tmp_path/'run', training_root=tmp_path/'training',
                 seed_start=10050, seeds=1, max_seed_attempts=1, baseline_repeats=2,
                 case_preset='baseline_smoke', check_only=False)
    value.update(changes)
    return SimpleNamespace(**value)


@pytest.mark.parametrize('change', [{'seeds':2}, {'max_seed_attempts':2}, {'baseline_repeats':3}, {'case_preset':'unknown'}])
def test_smoke_cannot_add_scenes_attempts_or_replays(tmp_path, change):
    with pytest.raises(ValueError): driver.selected_cases(args(tmp_path, **change), 'open_laptop')


def test_smoke_allowlist_and_default_case_compatibility(tmp_path):
    assert driver.selected_cases(args(tmp_path), 'open_laptop') == []
    with pytest.raises(ValueError): driver.selected_cases(args(tmp_path), 'adjust_bottle')
    legacy = SimpleNamespace()
    assert driver.selected_cases(legacy, 'open_laptop') == driver.default_cases()


@pytest.mark.parametrize('expert_raises', [False, True])
def test_failed_reference_keeps_initial_metadata_partial_samples_and_never_retries(tmp_path, monkeypatch, expert_raises):
    from scripts import collect_robotwin_clean50
    native, _ = environment('open_laptop', monkeypatch)
    receipt = initialize('open_laptop', native)
    native.check_success = lambda: False
    calls=[]
    def expert(env):
        calls.append(env)
        env._take_picture(); env.plan_success = True
        if expert_raises: raise RuntimeError('native planner raised')
    monkeypatch.setattr(collect_robotwin_clean50, 'play_expert', expert)
    monkeypatch.setattr(driver, 'install_base_frame_adapter', lambda *a, **k: {'audited':'fixture'})
    sample = {a:{'pose':[0.,0.,.8,1.,0.,0.,0.], 'gripper':1.} for a in ('left','right')}
    monkeypatch.setattr(driver, 'sample', lambda n: deepcopy(sample))
    target = tmp_path/'reference_attempt.json'
    with pytest.raises((ValueError, RuntimeError)):
        driver.capture_reference(lambda seed:native, {'task':'open_laptop'}, 10050, set(), attempt_path=target)
    audit = json.loads(target.read_text())
    assert audit['status']=='failed' and audit['success_metadata']==receipt
    assert audit['scene_sha256']==receipt['scene_before'] and len(audit['samples'])>=2
    assert calls==[native] and native.closed
    if expert_raises: assert audit['error']=='RuntimeError: native planner raised'
    else: assert audit['native_success'] is False


def mock_run(tmp_path, monkeypatch, *, first_valid=True, capture_error=None):
    from services.robotwin import acceptance, collection, runtime
    t=template(); t['task']='open_laptop'
    (tmp_path/'template.json').write_text(json.dumps(t))
    exclusions=dict(native_commit='native',excluded_training_seeds=[0],excluded_training_scene_sha256=['train'])
    monkeypatch.setattr(driver,'training_exclusions',lambda root:exclusions)
    monkeypatch.setattr(runtime,'source_commit',lambda:'native')
    monkeypatch.setattr(runtime,'initialize_runtime',lambda:None)
    monkeypatch.setattr(runtime,'native_config',lambda *a:{})
    monkeypatch.setattr(collection,'install_planner_cache',lambda:None)
    original_import = driver.importlib.import_module
    monkeypatch.setattr(driver.importlib,'import_module',lambda name:SimpleNamespace(open_laptop=object)
                        if name=='envs.open_laptop' else original_import(name))
    captures=[]; trials=[]
    def capture(factory, template, seed, scenes, **kw):
        captures.append((seed,kw))
        if capture_error: raise capture_error
        return reference(seed, template)
    def trial(output, manifest, planned, path, factory, baseline):
        trials.append(planned)
        raw=dict(native_success=first_valid or len(trials)>1, planner_success=True, realized=True,
                 execution_complete=True, seconds=.1)
        return {}, raw, {}
    monkeypatch.setattr(driver,'capture_reference',capture)
    monkeypatch.setattr(driver,'trial_record',trial)
    monkeypatch.setattr(driver,'verify_plan',lambda *a:None)
    monkeypatch.setattr(acceptance,'validate_manifest',lambda m:None)
    monkeypatch.setattr(acceptance,'evaluate_trial',lambda *a:dict(constraint_pass=True))
    monkeypatch.setattr(acceptance,'assess_evidence',lambda *a:dict(ready=False))
    return captures,trials


@pytest.mark.parametrize('first_valid,expected_count',[(False,1),(True,2)])
def test_second_baseline_is_conditional_and_first_failure_never_selects_new_seed(tmp_path, monkeypatch, first_valid, expected_count):
    captures,trials=mock_run(tmp_path,monkeypatch,first_valid=first_valid)
    assert driver.run(args(tmp_path)) == (0 if first_valid else 2)
    assert [r[0] for r in captures]==[10050]
    assert captures[0][1]['attempt_path']==tmp_path/'run/reference_attempts/seed10050.json'
    assert len(trials)==expected_count and all(p['variant_family']=='baseline' for p in trials)
    manifest=json.loads((tmp_path/'run/experiment_manifest.json').read_text())
    assert len(manifest['trial_plan'])==2 and manifest['purpose']=='native_metadata_smoke_not_acceptance'
    config=json.loads((tmp_path/'run/configuration.json').read_text())
    assert config['cases']==[] and config['case_preset']=='baseline_smoke'


def test_single_reference_failure_is_retained_without_baseline_or_seed_replacement(tmp_path, monkeypatch):
    captures,trials=mock_run(tmp_path,monkeypatch,capture_error=ValueError('expert did not succeed'))
    with pytest.raises(RuntimeError,match='Insufficient'): driver.run(args(tmp_path))
    assert [r[0] for r in captures]==[10050] and trials==[]
    failed=json.loads((tmp_path/'run/reference_rejections/seed10050.json').read_text())
    assert failed['reason']=='ValueError: expert did not succeed'
    with pytest.raises(ValueError,match='immutable'): driver.run(args(tmp_path))
    assert len(captures)==1


def test_factory_metadata_exception_still_leaves_attempt_without_native_actions(tmp_path):
    from services.robotwin.success_metadata import SuccessMetadataError
    def failed(seed): raise SuccessMetadataError('source mismatch')
    path=tmp_path/'attempt.json'
    with pytest.raises(SuccessMetadataError):
        driver.capture_reference(failed, {'task':'open_laptop'},10050,set(),attempt_path=path)
    audit=json.loads(path.read_text())
    assert audit['status']=='failed' and audit['samples']==[] and audit['native_actions_executed'] is False
    assert audit['error']=='SuccessMetadataError: source mismatch'
