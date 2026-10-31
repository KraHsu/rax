"""Complete numeric reference/sidecar tests. Synthetic traces are NOT native evidence."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from services.robotwin import reference_branch as guard
from services.robotwin.collection import scene_fingerprint
from scripts import validate_robotwin_constraint_acceptance as driver

FIXTURES = json.loads((Path(__file__).parent/'fixtures/reference_branches.json').read_text())


def fixture(name='place_mouse_pad/right', seed=None):
    item = deepcopy(FIXTURES[name])
    task, branch, registry = item['task'], item['branch_id'], item['registry']
    template = registry['entry']['template']
    keys = [b for (t, b), (_, status) in guard.BRANCHES.items() if t == task and status == 'active']
    start = 10000 + 100*keys.index(branch)
    spec = dict(version=1, policy=guard.POLICY, task=task, branch_id=branch,
                template_sha256=guard.template_hash(template), native_source_sha256=guard.SOURCE_SHA256[task],
                registry_path=registry['path'], registry_file_sha256=registry['file_sha256'], registry_entry=registry['entry'],
                seed_start=start, seeds=3, max_seed_attempts=30, baseline_repeats=2)
    count = max(e['frame'] for events in item['events'].values() for e in events)+3
    trajectory = {}
    for arm in ('left', 'right'):
        pose = np.tile([0., 0., .8, 1., 0., 0., 0.], (count, 1))
        if item['events'][arm]:
            pose[:, 2] += np.arange(count)*.0001
        grip = np.ones(count)
        for e in item['events'][arm]:
            grip[e['frame']:] = 0. if e['event']=='close' else 1.
        trajectory[arm] = dict(pose=pose.tolist(), gripper=grip.tolist())
    scene = item['initial_scene']
    ref = dict(trajectory=trajectory, initial_scene=scene,
        provenance=dict(task=task, seed=start if seed is None else seed, scene_sha256=scene_fingerprint(scene),
            mode='demo_clean', split='validation', source_episode_sha256=guard.canonical_hash(dict(scene=scene,trajectory=trajectory)),
            recording_method='native_expert_save_freq_trace_no_images'))
    return spec, template, ref


def rehash(ref):
    ref['provenance']['source_episode_sha256'] = guard.canonical_hash(dict(scene=ref['initial_scene'], trajectory=ref['trajectory']))


def seal(tmp_path, name='place_mouse_pad/right'):
    spec, template, ref = fixture(name)
    path, receipt = tmp_path/'reference.json', tmp_path/'receipt.json'
    path.write_text(json.dumps(ref))
    result = guard.seal_reference(path, receipt, spec, template)
    return spec, template, ref, path, receipt, result


def verify(values, **changes):
    spec, template, _, path, receipt, _ = values
    hashes = dict(expected_reference_sha256=guard.file_hash(path), expected_receipt_sha256=guard.file_hash(receipt))
    hashes.update(changes)
    return guard.verify_reference(path, receipt, spec, template, **hashes)


@pytest.mark.parametrize('name', list(FIXTURES))
def test_complete_reference_has_bound_T_and_recomputed_events(tmp_path, name):
    values = seal(tmp_path, name)
    result = verify(values)
    assert result['reference_events'] == FIXTURES[name]['events']
    assert result['action_frames']+1 == result['total_frames'] == len(values[2]['trajectory']['left']['pose'])
    assert result['reference_file_sha256'] == guard.file_hash(values[3])
    assert result['selection']['training_or_native_acceptance'] is False
    if name=='dustbin_static_mismatch':
        assert result['selection']['static_prediction_matches_reference'] is False


@pytest.mark.parametrize('field', ['total_frames', 'action_frames', 'branch_id', 'reference_events', 'source_episode_sha256'])
def test_coedited_receipt_hash_still_cannot_change_derived_fields(tmp_path, field):
    values = seal(tmp_path)
    receipt = json.loads(values[4].read_text())
    if field in ('total_frames','action_frames'):
        receipt[field] += 1
    elif field=='reference_events':
        receipt[field]['right'][0]['frame'] += 1
    else:
        receipt[field] = 'changed'
    values[4].write_text(json.dumps(receipt))
    with pytest.raises(guard.BranchIntegrityError, match='Receipt'):
        verify(values)


def test_reference_and_receipt_bytehash_are_both_mandatory(tmp_path):
    values = seal(tmp_path)
    for key in ('expected_reference_sha256','expected_receipt_sha256'):
        with pytest.raises(guard.BranchIntegrityError, match='byte hash'):
            verify(values, **{key:'0'*64})
    original = guard.file_hash(values[3])
    values[3].write_text(values[3].read_text()+' ')
    with pytest.raises(guard.BranchIntegrityError, match='byte hash'):
        verify(values, expected_reference_sha256=original)
    with pytest.raises(guard.BranchIntegrityError, match='immutable'):
        guard.seal_reference(values[3], values[4], values[0], values[1])


@pytest.mark.parametrize('change', ['one_arm_truncated','both_arms_truncated','nan','wrong_task','wrong_scene','role_metadata','supplied_events','gripper_metadata'])
def test_malformed_complete_reference_or_metadata_cannot_override_binding(change):
    spec, template, ref = fixture()
    if change in ('one_arm_truncated','both_arms_truncated'):
        for arm in (('left',) if change=='one_arm_truncated' else ('left','right')):
            for key in ('pose','gripper'): ref['trajectory'][arm][key].pop()
    elif change=='nan': ref['trajectory']['left']['pose'][2][0] = float('nan')
    elif change=='wrong_task': ref['provenance']['task']='place_bread_basket'
    elif change=='wrong_scene': ref['initial_scene']['actors'][0]['pose'][0] += .02
    elif change=='role_metadata': ref['provenance']['arm_tag']='left'
    elif change=='supplied_events': ref['reference_gripper_events']={'left':[],'right':[]}
    else: ref['trajectory']['right']['branch_id']='left'
    with pytest.raises(guard.BranchIntegrityError): guard.reference_admission(ref,spec,template)


def test_coedited_shortened_reference_with_old_receipt_T_rejected(tmp_path):
    values = list(seal(tmp_path))
    for arm in ('left','right'):
        for key in ('pose','gripper'): values[2]['trajectory'][arm][key].pop()
    rehash(values[2]);values[3].write_text(json.dumps(values[2]))
    receipt=json.loads(values[4].read_text());receipt['reference_file_sha256']=guard.file_hash(values[3])
    receipt['source_episode_sha256']=values[2]['provenance']['source_episode_sha256']
    values[4].write_text(json.dumps(receipt))
    with pytest.raises(guard.BranchIntegrityError,match='Receipt'):verify(values)


def test_legitimate_wrong_branch_is_a_bounded_rejection_not_template_switch():
    spec, template, _ = fixture('place_mouse_pad/right')
    _, _, left = fixture('place_mouse_pad/left', seed=spec['seed_start'])
    with pytest.raises(guard.ReferenceBranchMismatch, match='another'):
        guard.reference_admission(left,spec,template)


def test_exact_threshold_crossings_match_compiler_convention():
    spec,template,ref=fixture()
    events=FIXTURES['place_mouse_pad/right']['events']['right']
    for e in events:ref['trajectory']['right']['gripper'][e['frame']]=.5
    rehash(ref)
    result=guard.reference_admission(ref,spec,template)
    assert result['reference_events']['right']==events


@pytest.mark.parametrize('change', ['status','template','source','branch','budget','extra_metadata'])
def test_spec_identity_and_finite_bounds_fail_closed(change):
    spec,template,ref=fixture()
    if change=='status':spec['registry_entry']['status']='candidate'
    elif change=='template':spec['template_sha256']='0'*64
    elif change=='source':spec['native_source_sha256']='0'*64
    elif change=='branch':spec['branch_id']='left'
    elif change=='budget':spec['max_seed_attempts']=31
    else:spec['generated_arm_selector']='left'
    with pytest.raises(guard.BranchIntegrityError):guard.reference_admission(ref,spec,template)


def test_registry_and_native_sources_are_checked_without_gpu(tmp_path):
    spec,template,_=fixture()
    (tmp_path/'envs').mkdir()
    source=tmp_path/'envs/place_mouse_pad.py';source.write_text(FIXTURES['place_mouse_pad/right']['source'])
    registry=tmp_path/'registry.json';registry.write_text(json.dumps(spec['registry_entry']))
    spec.update(registry_path=str(registry),registry_file_sha256=guard.file_hash(registry))
    guard.validate_external_sources(spec,template,tmp_path)
    source.write_text(source.read_text()+'\n')
    with pytest.raises(guard.BranchIntegrityError,match='source drift'):guard.validate_external_sources(spec,template,tmp_path)


@pytest.mark.parametrize('cleanup_fails',[False,True])
def test_capture_rejected_branch_keeps_complete_attempt_and_closes_native(tmp_path,monkeypatch,cleanup_fails):
    from scripts import collect_robotwin_clean50
    spec,template,_=fixture('place_mouse_pad/right')
    _,_,ref=fixture('place_mouse_pad/left',seed=spec['seed_start'])
    samples=[{a:{'pose':ref['trajectory'][a]['pose'][i],'gripper':ref['trajectory'][a]['gripper'][i]} for a in ('left','right')}
             for i in range(len(ref['trajectory']['left']['pose']))]
    iterator=iter(samples);native=SimpleNamespace(plan_success=True,closed=False,check_success=lambda:True)
    def close():
        native.closed=True
        if cleanup_fails:raise RuntimeError('cleanup fixture failed')
    native.close_env=close
    monkeypatch.setattr(driver,'initial_scene',lambda _:ref['initial_scene'])
    monkeypatch.setattr(driver,'sample',lambda _:next(iterator))
    monkeypatch.setattr(driver,'install_base_frame_adapter',lambda *a,**kw:{'fixture':True})
    def play(env):
        for _ in range(len(samples)-2):env._take_picture()
    monkeypatch.setattr(collect_robotwin_clean50,'play_expert',play)
    path=tmp_path/'attempt.json'
    with pytest.raises(guard.BranchIntegrityError if cleanup_fails else guard.ReferenceBranchMismatch):
        driver.capture_reference(lambda _:native,template,spec['seed_start'],set(),attempt_path=path,branch_spec=spec)
    attempt=json.loads(path.read_text())
    assert native.closed and attempt['native_success'] is True and attempt['status']=='failed'
    assert len(attempt['samples'])==len(samples)
    assert len(attempt['branch_reference']['trajectory']['left']['pose'])==len(samples)
    if cleanup_fails:assert attempt['cleanup_error']=='RuntimeError: cleanup fixture failed'


def test_trial_checks_receipt_before_native_or_output_directory(tmp_path):
    spec,template,_=fixture()
    manifest=dict(branch_spec=spec,template=template,reference_artifacts=[])
    with pytest.raises(guard.BranchIntegrityError,match='unbound'):
        driver.trial_record(tmp_path,manifest,{'seed':spec['seed_start']},tmp_path/'reference.json',
                            lambda _:pytest.fail('native must not start'),None)
    assert not (tmp_path/'trials').exists()


def test_final_sealed_reference_recompiles_exactly_and_label_tamper_is_rejected(tmp_path):
    spec,template,ref=fixture()
    ref['bundle'],ref['template_audit']=driver.reference_bundle(template,ref['trajectory'],spec['seed_start'],ref['provenance']['scene_sha256'])
    path,receipt=tmp_path/'reference.json',tmp_path/'receipt.json'
    path.write_text(json.dumps(ref));guard.seal_reference(path,receipt,spec,template)
    entry=dict(seed=spec['seed_start'],path=path.name,sha256=guard.file_hash(path),
        branch_receipt_path=receipt.name,branch_receipt_sha256=guard.file_hash(receipt))
    manifest=dict(branch_spec=spec,template=template)
    driver.verify_branch_entry(manifest,entry,tmp_path,template)
    ref['bundle']['constraints'][0]['weight']=2.
    path.write_text(json.dumps(ref));receipt.unlink();guard.seal_reference(path,receipt,spec,template)
    entry.update(sha256=guard.file_hash(path),branch_receipt_sha256=guard.file_hash(receipt))
    with pytest.raises(guard.BranchIntegrityError,match='labels differ'):
        driver.verify_branch_entry(manifest,entry,tmp_path,template)


def mock_branch_run(tmp_path,monkeypatch,*,capture_error=None,first_baseline_valid=False):
    from services.robotwin import acceptance,collection,runtime
    spec,template,_=fixture()
    (tmp_path/'template.json').write_text(json.dumps(template));(tmp_path/'spec.json').write_text(json.dumps(spec))
    args=SimpleNamespace(template=tmp_path/'template.json',branch_spec=tmp_path/'spec.json',output=tmp_path/'run',
        training_root=tmp_path/'training',seed_start=spec['seed_start'],seeds=3,max_seed_attempts=30,baseline_repeats=2,check_only=False)
    monkeypatch.setenv('ROBOTWIN_PATH',str(tmp_path))
    monkeypatch.setattr(driver,'validate_external_sources',lambda *a:None)
    monkeypatch.setattr(driver,'training_exclusions',lambda _:dict(native_commit='native',excluded_training_seeds=[],excluded_training_scene_sha256=[]))
    monkeypatch.setattr(runtime,'source_commit',lambda:'native');monkeypatch.setattr(runtime,'initialize_runtime',lambda:None)
    monkeypatch.setattr(runtime,'native_config',lambda *a:{})
    monkeypatch.setattr(collection,'install_planner_cache',lambda:None)
    original_import=driver.importlib.import_module
    monkeypatch.setattr(driver.importlib,'import_module',lambda name:SimpleNamespace(place_mouse_pad=object)
        if name=='envs.place_mouse_pad' else original_import(name))
    captures=[];trials=[]
    def capture(factory,template,seed,scenes,**kw):
        captures.append((seed,kw))
        if capture_error:raise capture_error
        _,_,ref=fixture(seed=seed)
        ref['initial_scene']['actors'][0]['pose'][0]+=(seed-spec['seed_start'])*.001
        ref['provenance']['scene_sha256']=scene_fingerprint(ref['initial_scene']);rehash(ref)
        ref['bundle'],ref['template_audit']=driver.reference_bundle(template,ref['trajectory'],seed,ref['provenance']['scene_sha256'])
        ref.update(replay_adapter={'fixture':True},replay_adapter_applied=False)
        return ref
    def trial(output,manifest,planned,path,factory,baseline):
        trials.append(planned)
        return {},dict(native_success=first_baseline_valid,planner_success=True,realized=True,execution_complete=True,seconds=.1),{}
    monkeypatch.setattr(driver,'capture_reference',capture);monkeypatch.setattr(driver,'trial_record',trial)
    monkeypatch.setattr(acceptance,'validate_manifest',lambda _:None)
    monkeypatch.setattr(acceptance,'evaluate_trial',lambda *a:dict(constraint_pass=True))
    monkeypatch.setattr(acceptance,'assess_evidence',lambda *a:dict(ready=False))
    return args,captures,trials


def test_branch_bad_baseline_never_retries_or_selects_replacement(tmp_path,monkeypatch):
    args,captures,trials=mock_branch_run(tmp_path,monkeypatch)
    assert driver.run(args)==2
    assert len(captures)==3 and len(trials)==1 and trials[0]['case']['id']=='baseline_0'
    assert all(c[1]['attempt_path'].parent.name=='reference_attempts' and c[1]['branch_spec']['branch_id']=='right' for c in captures)
    manifest=json.loads((args.output/'experiment_manifest.json').read_text())
    assert len(manifest['trial_plan'])==60 and len(manifest['reference_artifacts'])==3
    for entry in manifest['reference_artifacts']:assert 'branch_receipt_sha256' in entry
    with pytest.raises(guard.BranchIntegrityError,match='immutable'):driver.run(args)
    assert len(captures)==3 and len(trials)==1


@pytest.mark.parametrize('fatal',[False,True])
def test_wrong_branch_scan_is_bounded_but_integrity_failure_never_changes_seed(tmp_path,monkeypatch,fatal):
    error=guard.BranchIntegrityError('native cleanup failed') if fatal else guard.ReferenceBranchMismatch('other branch')
    args,captures,trials=mock_branch_run(tmp_path,monkeypatch,capture_error=error)
    with pytest.raises(guard.BranchIntegrityError if fatal else RuntimeError):driver.run(args)
    assert len(captures)==(1 if fatal else 30) and not trials
    rejects=list((args.output/'reference_rejections').glob('*.json'))
    assert len(rejects)==(0 if fatal else 30)
