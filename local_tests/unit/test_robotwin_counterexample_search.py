"""Discovery cannot confuse label admissibility with physical/native outcomes."""
import copy
from types import SimpleNamespace

import pytest

from scripts import search_robotwin_constraint_counterexamples as search
from services.robotwin.acceptance import annotation_policy_fingerprint, file_hash, json_hash
from services.robotwin.acceptance_replay import tracking_report
from services.robotwin.collection import atomic_json, scene_fingerprint
from local_tests.unit.test_robotwin_acceptance import reference as reference_fixture, template


def reference():
    value = reference_fixture(10000, template())
    value['initial_scene'] = {'actors': [dict(name='object', pose=[0., .1, .8, 1., 0., 0., 0.])]}
    value['provenance']['scene_sha256'] = scene_fingerprint(value['initial_scene'])
    value['provenance']['source_episode_sha256'] = json_hash(dict(scene=value['initial_scene'], trajectory=value['trajectory']))
    return value


def native_result(ref, commanded, actual=None, *, success=False, baseline=None, bad_planner=False):
    actual = copy.deepcopy(commanded if actual is None else actual)
    length = len(commanded['left']['pose'])
    planners = [dict(arm=arm, frame=i, status='Success') for i in range(1,length) for arm in ('left','right')]
    if bad_planner:
        planners[-1]['status'] = 'Failure'
    track = tracking_report(actual, commanded, baseline=baseline,
                            reference=ref['trajectory'] if baseline is not None else None)
    raw = dict(execution_complete=True, planner_success=not bad_planner, planner_results=planners,
               native_success=success, ever_success=success, final_success=success,
               replay_adapter=ref['replay_adapter'], replay_adapter_applied=True,
               realized=track['realized'], tracking=track)
    return raw, actual


def early_close(ref):
    candidate = copy.deepcopy(ref['trajectory'])
    candidate['left']['gripper'][1:] = [0.]*(len(candidate['left']['gripper'])-1)
    return dict(trajectory=candidate, source='crafted', family='early_close')


def test_realized_allowed_failure_is_primary_and_strong_counterexample():
    ref = reference(); candidate = early_close(ref)
    raw, actual = native_result(ref,candidate['trajectory'],baseline=ref['trajectory'])
    result = search.classify_trial(ref,candidate,raw,actual,baseline=ref['trajectory'])
    assert result['commanded_constraint_pass'] and result['actual_constraint_pass']
    assert result['is_counterexample'] and result['strong_counterexample']
    assert not result['controller_failure']


def test_planner_failure_or_unrealized_candidate_is_never_task_negative():
    ref = reference(); candidate = early_close(ref)
    raw, actual = native_result(ref,candidate['trajectory'],bad_planner=True,baseline=ref['trajectory'])
    result = search.classify_trial(ref,candidate,raw,actual,baseline=ref['trajectory'])
    assert result['controller_failure'] and not result['is_counterexample']
    actual = copy.deepcopy(candidate['trajectory']); actual['left']['pose'][2][0] += .02
    raw, actual = native_result(ref,candidate['trajectory'],actual,baseline=ref['trajectory'])
    result = search.classify_trial(ref,candidate,raw,actual,baseline=ref['trajectory'])
    assert not result['realized'] and not result['is_counterexample']
    incomplete = dict(raw,execution_complete=False,native_success=False,realized=False)
    result = search.classify_trial(ref,candidate,incomplete,{'left':{},'right':{}})
    assert result['native_success'] is None and not result['is_counterexample']


def test_command_allowed_but_actual_outside_is_not_strong_counterexample():
    ref = reference(); candidate = early_close(ref)
    for p in candidate['trajectory']['left']['pose'][1:]: p[0] += .0048
    actual = copy.deepcopy(candidate['trajectory'])
    for p in actual['left']['pose'][1:]: p[0] += .0012
    raw, actual = native_result(ref,candidate['trajectory'],actual,baseline=ref['trajectory'])
    result = search.classify_trial(ref,candidate,raw,actual,baseline=ref['trajectory'])
    assert result['realized'] and result['commanded_constraint_pass']
    assert not result['actual_constraint_pass']
    assert result['is_counterexample'] and result['command_only_counterexample']
    assert not result['strong_counterexample']


def test_forged_realization_or_changed_initial_state_rejected():
    ref = reference(); candidate = early_close(ref)
    raw, actual = native_result(ref,candidate['trajectory'],baseline=ref['trajectory'])
    with pytest.raises(ValueError,match='realization'):
        search.classify_trial(ref,candidate,dict(raw,realized=False),actual,baseline=ref['trajectory'])
    actual['left']['pose'][0][0] += .001
    with pytest.raises(ValueError,match='initial'):
        search.classify_trial(ref,candidate,raw,actual,baseline=ref['trajectory'])


def test_partition_and_holdout_release_prevent_leakage():
    split=dict(version=1,task='adjust_bottle',dev_seeds=[10000],holdout_seeds=[11000])
    kwargs=dict(task='adjust_bottle',scene_seed=10000,exclusions={'excluded_training_seeds':[1,2]},
                template_sha256='a'*64,annotation_policy_sha256='b'*64)
    search.validate_split(split,'dev',**kwargs)
    with pytest.raises(ValueError,match='disjoint'):
        search.validate_split(dict(split,holdout_seeds=[10000]),'dev',**kwargs)
    kwargs['scene_seed']=11000
    with pytest.raises(ValueError,match='release'):
        search.validate_split(split,'holdout',**kwargs)
    release=dict(version=1,task=split['task'],template_sha256='a'*64,annotation_policy_sha256='b'*64,
                 split_plan_sha256=json_hash(split),frozen_unix=1.,development_report_sha256='c'*64)
    search.validate_split(split,'holdout',release=release,**kwargs)
    with pytest.raises(ValueError,match='release'):
        search.validate_split(split,'holdout',release=dict(release,template_sha256='d'*64),**kwargs)


def test_real_model_import_requires_source_identity_and_exact_N(tmp_path):
    ref=reference(); row=early_close(ref)
    row.update(source='model',family='model_closed_loop',candidate_id='policy0',
               model_provenance=dict(source_kind='model_closed_loop',checkpoint_sha256='a'*64,source_run_sha256='b'*64,full_policy_rollout=True))
    value=dict(version=1,task='adjust_bottle',scene_seed=10000,scene_sha256=ref['provenance']['scene_sha256'],
               alignment=search.ALIGNMENT,action_contract=ref['bundle']['action_contract'],candidates=[row])
    path=tmp_path/'models.json';atomic_json(path,value)
    assert search.load_model_inputs(path,ref)[0]['model_provenance']['source_kind']=='model_closed_loop'
    value['candidates'][0]['trajectory']['left']['pose'].pop();atomic_json(path,value)
    with pytest.raises(ValueError): search.load_model_inputs(path,ref)
    value['candidates'][0]['trajectory']=early_close(ref)['trajectory']
    value['candidates'][0]['model_provenance'].pop('checkpoint_sha256');atomic_json(path,value)
    with pytest.raises(ValueError,match='provenance'):search.load_model_inputs(path,ref)


def test_holdout_replays_rejected_candidates_under_frozen_generator():
    controls = dict(random_seed=7, candidate_budget=6, baseline_repeats=2, include_outside_controls=True)
    generator = dict(version=1, generator_sha256='a'*64, configuration=controls, sources={})
    config = dict(controls, split='holdout', split_plan={'holdout_generator':generator},
                  holdout_release=dict(holdout_generator_sha256=json_hash(generator), frozen_unix=1.),
                  feedback_inputs_sha256=None)
    search.validate_holdout_configuration(config,'a'*64,created_unix=2.)
    for changed in (dict(include_outside_controls=False),dict(random_seed=8),dict(feedback_inputs_sha256='b'*64)):
        with pytest.raises(ValueError):
            search.validate_holdout_configuration(dict(config,**changed),'a'*64,created_unix=2.)
    with pytest.raises(ValueError,match='predate'):
        search.validate_holdout_configuration(config,'a'*64,created_unix=.5)
    with pytest.raises(ValueError,match='settings'):
        search.validate_holdout_configuration(config,'c'*64,created_unix=2.)
    candidates=[dict(candidate_id='inside',source='crafted',family='test',commanded_constraint_pass=True),
                dict(candidate_id='outside',source='crafted',family='test',commanded_constraint_pass=False)]
    assert len(search.make_trial_plan(candidates,2))==3
    assert len(search.make_trial_plan(candidates,2,include_outside_controls=True))==4


def campaign(tmp_path,monkeypatch):
    source=reference(); src=tmp_path/'source';src.mkdir()
    atomic_json(src/'reference.json',source)
    t=template()
    old_manifest=dict(version=1,task=t['task'],mode='demo_clean',split='validation',template=t,
                      template_sha256=search.template_hash(t),native_commit='a'*40,
                      annotation_policy_sha256='c'*64,alignment=search.ALIGNMENT,native_success_rule=search.SUCCESS_RULE,
                      replay_adapter=source['replay_adapter'],reference_artifacts=[dict(seed=10000,path='reference.json',sha256=file_hash(src/'reference.json'))])
    atomic_json(src/'manifest.json',old_manifest);atomic_json(src/'template.json',t)
    atomic_json(src/'split.json',dict(version=1,task=t['task'],dev_seeds=[10000],holdout_seeds=[11000]))
    exclusions=dict(excluded_training_seeds=[1,2],excluded_training_scene_sha256=['training-scene'],native_commit='a'*40,source_episodes=2)
    monkeypatch.setattr(search,'training_exclusions',lambda *_:exclusions)
    args=SimpleNamespace(reference=src/'reference.json',reference_manifest=src/'manifest.json',template=src/'template.json',
                         training_root=src,output=tmp_path/'out',split_plan=src/'split.json',split='dev',holdout_release=None,
                         random_seed=7,candidate_budget=6,baseline_repeats=2,include_outside_controls=False,
                         model_candidates=None,feedback=None,check_only=True)
    manifest=search.prepare_campaign(args)
    return args,manifest,source


def test_old_source_policy_preserved_and_current_labels_frozen(tmp_path,monkeypatch):
    args,manifest,source=campaign(tmp_path,monkeypatch)
    assert manifest['source_annotation_policy_sha256']=='c'*64
    assert manifest['annotation_policy_sha256']==annotation_policy_fingerprint()
    assert (args.output/'source_reference.json').read_bytes()==args.reference.read_bytes()
    report=search.verify_campaign(args.output)
    assert not report['complete'] and report['acceptance_ready'] is False
    assert len(manifest['trial_plan']) > 2
    args.candidate_budget += 1
    with pytest.raises(ValueError,match='Frozen'):
        search.prepare_campaign(args)


def execute_fake(root,manifest,planned,monkeypatch,*,success):
    ref=search.checked_artifact(root,manifest['reference'])
    native=SimpleNamespace(close_env=lambda:None)
    monkeypatch.setattr(search,'initial_scene',lambda _:ref['initial_scene'])
    monkeypatch.setattr(search,'install_base_frame_adapter',lambda _:ref['replay_adapter'])
    monkeypatch.setattr(search,'sample',lambda _: {arm:dict(pose=ref['trajectory'][arm]['pose'][0],gripper=ref['trajectory'][arm]['gripper'][0]) for arm in ('left','right')})
    def fake_replay(_,commanded):
        raw,actual=native_result(ref,commanded,success=success)
        return dict(raw,actual=actual)
    monkeypatch.setattr(search,'replay',fake_replay)
    search.execute_trial(root,manifest,planned,lambda seed:native)


def test_committed_trials_reassessed_and_hash_changes_rejected(tmp_path,monkeypatch):
    args,manifest,_=campaign(tmp_path,monkeypatch);root=args.output
    for planned in manifest['trial_plan']:
        execute_fake(root,manifest,planned,monkeypatch,success=planned['kind']=='baseline')
    report=search.verify_campaign(root)
    assert report['complete'] and report['counts']['strong_counterexamples'] > 0
    assert report['counts']['counterexamples']==report['counts']['candidates']
    assert not report['acceptance_ready']
    path=root/'trials/candidate_0000/trajectory.json'
    if not path.exists(): path=next((root/'trials').glob('candidate_*/trajectory.json'))
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='hash'):
        search.verify_campaign(root)


def test_feedback_requires_complete_dev_true_inside_set_failures(tmp_path,monkeypatch):
    args,manifest,_=campaign(tmp_path,monkeypatch);root=args.output
    for planned in manifest['trial_plan']:
        execute_fake(root,manifest,planned,monkeypatch,success=planned['kind']=='baseline')
    candidate=next(p for p in manifest['trial_plan'] if p['kind']=='candidate')
    request=tmp_path/'feedback.json';atomic_json(request,dict(version=1,campaigns=[dict(campaign_root=str(root),trial_ids=[candidate['trial_id']])]))
    ref=search.checked_artifact(root,manifest['reference'])
    parents,receipts=search.load_feedback(request,ref)
    assert len(parents)==len(receipts)==1 and parents[0]['native_success'] is False
    request.write_text(request.read_text().replace(candidate['trial_id'],'baseline_0'))
    with pytest.raises(ValueError,match='inside-set'):
        search.load_feedback(request,ref)
