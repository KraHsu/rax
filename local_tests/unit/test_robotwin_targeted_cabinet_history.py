"""New package integration only; existing 286 observer checks stay separately frozen."""
from copy import deepcopy
import json
import pytest
from scripts import observe_robotwin_expert_failure as driver


def fixture(tmp_path):
    names=['fresh_metadata_native_smoke_v1/runs/put_object_cabinet/reference_attempts/seed10050.json','cabinet_expert_observation_v1/runs/put_object_cabinet/expert_attempt.json']
    d=dict(task='put_object_cabinet',targeted_observation_version=driver.TARGETED_VERSION,planner_solver_details=True,external_files={},input_files={})
    h=dict(previous_cabinet_initializations=2,new_cabinet_initializations_max=1,cumulative_cabinet_initializations_max=3,sources=[])
    for ordinal,name in enumerate(names,1):
        p=tmp_path/('old'+str(ordinal)+'.json');x=dict(task='put_object_cabinet',seed=10050,scene_sha256=driver.TASKS['put_object_cabinet']['scene'],status='failed',planner_success=False,native_success=None,original_ordinal=ordinal)
        p.write_text(json.dumps(x));sha=driver.file_hash(p);path='/old/'+name
        d['input_files'][p.name]=sha;d['external_files'][path]=sha
        h['sources'].append(dict(ordinal=ordinal,source_path=path,input_file=p.name,sha256=sha))
    d['previous_failure']=p.name
    return d,h


def test_new_history_accepts_exact_two_originals_plus_one(tmp_path):
    d,h=fixture(tmp_path);before={p.name:p.read_bytes()for p in tmp_path.iterdir()}
    driver.validate_targeted_history(tmp_path,d,h)
    assert {p.name:p.read_bytes()for p in tmp_path.iterdir()}==before


@pytest.mark.parametrize('change',['no_optin','task','version','previous','budget','replay_history','one_source','same_source','ordinal','latest','input_hash','source_hash','native_label'])
def test_targeted_history_rejects_budget_identity_or_source_changes(tmp_path,change):
    d,h=fixture(tmp_path)
    if change=='no_optin':d['planner_solver_details']=False
    elif change=='task':d['task']='place_dual_shoes'
    elif change=='version':d['targeted_observation_version']='anything'
    elif change=='previous':h['previous_cabinet_initializations']=1
    elif change=='budget':h['new_cabinet_initializations_max']=True
    elif change=='replay_history':h['cumulative_cabinet_initializations_max']=4
    elif change=='one_source':h['sources'].pop()
    elif change=='same_source':h['sources'][1]=deepcopy(h['sources'][0])
    elif change=='ordinal':h['sources'][0]['ordinal']=2
    elif change=='latest':d['previous_failure']=h['sources'][0]['input_file']
    elif change=='input_hash':d['input_files']['old1.json']='0'*64
    elif change=='source_hash':d['external_files'][h['sources'][0]['source_path']]='0'*64
    else:
        p=tmp_path/'old2.json';x=json.loads(p.read_text());x['native_success']=False;p.write_text(json.dumps(x));sha=driver.file_hash(p)
        e=h['sources'][1];e['sha256']=sha;d['input_files'][e['input_file']]=sha;d['external_files'][e['source_path']]=sha
    with pytest.raises(ValueError):driver.validate_targeted_history(tmp_path,d,h)


from test_robotwin_failure_observation import terminal_native
from services.robotwin.success_metadata import MARKER,fresh_physical_state


def state_fixture(monkeypatch):
    n,_,_=terminal_native(monkeypatch)
    receipt=n.__dict__[MARKER]
    d=dict(task='put_object_cabinet',success_metadata_policy=receipt['policy'],expected_scene_sha256=receipt['scene_before'])
    return n,d,receipt


def test_initial_state_reuses_existing_metadata_without_any_physical_getter(monkeypatch,tmp_path):
    n,d,receipt=state_fixture(monkeypatch)
    before=deepcopy(n.__dict__[MARKER])
    def forbid(*a,**k):raise AssertionError('No additional physical access')
    n.scene.get_all_actors=forbid;n.scene.get_all_articulations=forbid;n.scene.step=forbid
    driver.retain_targeted_initial_state(tmp_path,n,d,d['expected_scene_sha256'])
    driver.validate_targeted_initial_state(tmp_path,d,dict(success_metadata=receipt),required=True)
    assert n.__dict__[MARKER]==before


@pytest.mark.parametrize('change',['physical','receipt','missing','attempt','scene'])
def test_initial_state_and_attempt_binding_rejects_tamper(monkeypatch,tmp_path,change):
    n,d,r=state_fixture(monkeypatch);driver.retain_targeted_initial_state(tmp_path,n,d,d['expected_scene_sha256'])
    attempt=dict(success_metadata=r)
    if change=='physical':driver.atomic_json(tmp_path/'expert_initial_physical_state.json',{})
    elif change=='receipt':
        x=deepcopy(r);x['fresh_state_sha256']='0'*64;driver.atomic_json(tmp_path/'expert_initial_success_metadata.json',x)
    elif change=='missing':(tmp_path/'expert_initial_physical_state.json').unlink()
    elif change=='attempt':attempt['success_metadata']=None
    else:d['expected_scene_sha256']='0'*64
    with pytest.raises(ValueError):driver.validate_targeted_initial_state(tmp_path,d,attempt,required=True)


def test_actual_binding_failure_consumes_only_init_retains_initial_state(monkeypatch,tmp_path):
    n,d,r=state_fixture(monkeypatch);calls=[];closed=[];before=fresh_physical_state(n)
    n.close_env=lambda:closed.append(True)
    d.update(kind=driver.KIND,seed=10050,targeted_observation_version=driver.TARGETED_VERSION,planner_solver_details=True,planner_policy=driver.planner_policy(solver_details=True),**driver.FLAGS)
    def factory(seed):calls.append(seed);return n
    def reject(*a,**k):raise ValueError('actual source/type binding incompatibility')
    monkeypatch.setattr(driver,'ExpertPlannerObserver',reject)
    driver.atomic_json(tmp_path/'observation_manifest.json',dict(kind=driver.KIND,design=d,design_sha256='d'*64,**driver.FLAGS))
    result=driver.run_one(tmp_path,d,dict(task='put_object_cabinet'),dict(excluded_training_scene_sha256=[]),factory)
    assert calls==[10050]and closed==[True]and result['native_initializations_started']==1
    assert result['expert_attempts_started']==0 and result['legacy_replays']==0 and result['status']=='observation_incomplete'
    assert 'binding incompatibility'in result['original_capture_error']['message']
    assert fresh_physical_state(n)==before
    assert (tmp_path/'expert_initial_physical_state.json').exists()
    driver.verify_output(tmp_path,'d'*64,d)
