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
