from copy import deepcopy

import pytest

from scripts import observe_robotwin_expert_failure as driver
from scripts import collect_robotwin_clean50
from services.robotwin import expert_terminal_evidence as shoes
from test_robotwin_expert_terminal_evidence import environment as shoe_environment
from test_robotwin_failure_observation import F
from services.robotwin.collection import initial_scene,scene_fingerprint


def design(task,scene):
    return dict(kind=driver.KIND,task=task,seed=driver.TASKS[task]['seed'],expected_scene_sha256=scene,
        terminal_policy=shoes.policy(task)if task=='place_dual_shoes'else driver.cabinet.policy(F/'cabinet_model_data.json'),
        planner_policy=None if task=='place_dual_shoes'else driver.planner_policy(),**driver.FLAGS)


def observed_shoe(tmp_path,monkeypatch,*,failure=False,expert_error=False,observe_error=False):
    n,_=shoe_environment(monkeypatch);n.left_shoe.pose.p[0]=.06 if failure else 0.
    scene=scene_fingerprint(initial_scene(n));d=design('place_dual_shoes',scene);out=tmp_path/'out';out.mkdir()
    driver.atomic_json(out/'observation_manifest.json',dict(kind=driver.KIND,design=d,design_sha256='d'*64,**driver.FLAGS))
    calls=[]
    def factory(seed):calls.append(seed);return n
    def expert(native):
        calls.append('expert');native._take_picture()
        if expert_error:raise RuntimeError('original expert exception')
    monkeypatch.setattr(collect_robotwin_clean50,'play_expert',expert)
    monkeypatch.setattr(driver.legacy,'install_base_frame_adapter',lambda *a,**k:{})
    sample={a:dict(pose=[0.,0.,.8,1.,0.,0.,0.],gripper=1.)for a in ('left','right')}
    monkeypatch.setattr(driver.legacy,'sample',lambda n:deepcopy(sample))
    monkeypatch.setattr(driver.legacy,'reference_bundle',lambda *a:({},{}))
    if observe_error:
        def error(*a,**k):raise OSError('terminal observer failed')
        monkeypatch.setattr(shoes,'observe',error)
    r=driver.run_one(out,d,dict(task='place_dual_shoes'),dict(excluded_training_scene_sha256=[]),factory)
    return out,d,r,calls,n


@pytest.mark.parametrize('failure,expert_error',[(False,False),(True,False),(False,True)])
def test_one_expert_no_replay_and_original_failure_retained(tmp_path,monkeypatch,failure,expert_error):
    out,d,r,calls,n=observed_shoe(tmp_path,monkeypatch,failure=failure,expert_error=expert_error)
    assert calls==[10000,'expert'] and n.closed
    assert r['native_initializations_started']==r['expert_attempts_started']==1 and r['legacy_replays']==0
    assert r['status']=='observation_complete'
    result=driver.verify_output(out,'d'*64,d);assert result['original_native_success']==(None if expert_error else not failure)
    if expert_error:assert result['original_capture_error']['message']=='original expert exception'
    if failure:assert result['original_capture_error']['message']=='Reference native expert did not succeed'
    assert not any('replay'in p.name or 'baseline'in p.name for p in out.iterdir())


def test_observer_error_keeps_primary_error_and_no_second_expert(tmp_path,monkeypatch):
    out,d,r,calls,n=observed_shoe(tmp_path,monkeypatch,expert_error=True,observe_error=True)
    assert calls==[10000,'expert']and n.closed and r['status']=='observation_incomplete'
    assert r['original_capture_error']['message']=='original expert exception'
    a=driver.read(out/'expert_attempt.json');assert 'terminal observer failed'in a['terminal_observation_error']
    driver.verify_output(out,'d'*64,d)


@pytest.mark.parametrize('tamper',['design','labels','budget','component','missing','orphan','completion'])
def test_rehashed_or_orphan_evidence_does_not_pass(tmp_path,monkeypatch,tamper):
    out,d,r,calls,n=observed_shoe(tmp_path,monkeypatch,failure=True)
    if tamper=='orphan':driver.atomic_json(out/'extra_result.json',{} )
    elif tamper=='missing':(out/'expert_terminal_evidence.json').unlink();driver.commit_index(out)
    else:
        name='run.json'if tamper in('labels','budget','completion')else'observation_manifest.json'if tamper=='design'else'expert_terminal_evidence.json'
        x=driver.read(out/name)
        if tamper=='design':x['design']['seed']=10001
        elif tamper=='labels':x['original_native_success']=True
        elif tamper=='budget':x['native_initializations_started']=2
        elif tamper=='completion':x['status']='observation_incomplete'
        else:x['component_result']['components']['left_xy']['passed']=True
        driver.atomic_json(out/name,x);driver.commit_index(out)
    with pytest.raises((ValueError,FileNotFoundError)):driver.verify_output(out,'d'*64,d)


def test_wrong_scene_prevents_expert_and_terminal_and_retains_failure(tmp_path,monkeypatch):
    n,_=shoe_environment(monkeypatch);d=design('place_dual_shoes','x'*64);out=tmp_path/'out';out.mkdir()
    driver.atomic_json(out/'observation_manifest.json',dict(kind=driver.KIND,design=d,design_sha256='d'*64,**driver.FLAGS))
    calls=[]
    def forbidden(*a,**k):raise AssertionError('expert cannot start')
    monkeypatch.setattr(collect_robotwin_clean50,'play_expert',forbidden)
    result=driver.run_one(out,d,dict(task='place_dual_shoes'),dict(excluded_training_scene_sha256=[]),lambda seed:(calls.append(seed)or n))
    assert calls==[10000]and n.closed and result['status']=='observation_incomplete'
    assert 'Initial scene differs'in result['original_capture_error']['message']
    assert not(out/'expert_terminal_evidence.json').exists()
    driver.verify_output(out,'d'*64,d)
