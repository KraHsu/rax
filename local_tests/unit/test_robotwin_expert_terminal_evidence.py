"""CPU only: original native AST predicate and read-only final evidence."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from services.robotwin import expert_terminal_evidence as terminal
from services.robotwin.success_metadata import fresh_physical_state
from scripts import diagnose_robotwin_capture_timing as timing
from scripts import validate_robotwin_constraint_acceptance as capture
from test_robotwin_success_metadata import Entity

FIXTURES=Path(__file__).parents[1]/'fixtures/robotwin_terminal_evidence'


def native_class(path, name, monkeypatch, *, methods=None, **globals_):
    module=ModuleType('terminal_test_'+name);module.__file__=str(path);module.__dict__.update(globals_)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if getattr(n,'name',None)==name)
    if methods is not None:
        node.bases=[];node.body=[n for n in node.body if getattr(n,'name',None) in methods]
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),module.__dict__)
    return getattr(module,name),module


def environment(monkeypatch):
    for name,file in [(terminal.TASK,'place_dual_shoes.py'),('robot','robot.py'),('base_task','_base_task.py')]:
        assert hashlib.sha256((FIXTURES/file).read_bytes()).hexdigest()==terminal.NATIVE_SHA256[name]
    Base,_=native_class(FIXTURES/'_base_task.py','Base_Task',monkeypatch,
        methods={'is_left_gripper_open','is_right_gripper_open'})
    Robot,_=native_class(FIXTURES/'robot.py','Robot',monkeypatch,
        methods={'is_left_gripper_open','is_right_gripper_open','get_left_gripper_val','get_right_gripper_val'})
    Native,module=native_class(FIXTURES/'place_dual_shoes.py',terminal.TASK,monkeypatch,Base_Task=Base,np=np)
    n=Native();n.robot=Robot();n.robot.left_gripper=n.robot.right_gripper=[object()]
    n.robot.left_gripper_val=n.robot.right_gripper_val=1.
    q=[.5,.5,-.5,-.5]
    n.left_shoe=Entity('041_shoe',[0.,-.17,.65],q);n.right_shoe=Entity('041_shoe',[0.,-.09,.65],q)
    n.shoe_box=Entity('007_shoe-box',[0.,-.13,.64],q)
    def forbidden(*a,**k):raise AssertionError('Observation invoked get_info/control/step')
    n.scene=SimpleNamespace(get_all_actors=lambda:[n.left_shoe,n.right_shoe,n.shoe_box],
        get_all_articulations=lambda:[],get_timestep=lambda:.004,step=forbidden)
    n.closed=False;n.close_env=lambda:setattr(n,'closed',True);n.plan_success=True
    for name in ('get_info','play_once','take_action','move','delay'):setattr(n,name,forbidden)
    return n,module


@pytest.mark.parametrize('failure',[False,True])
def test_source_native_components_and_final_snapshot_unchanged(monkeypatch,failure):
    n,_=environment(monkeypatch)
    if failure:n.left_shoe.pose.p[0]=.06
    original=n.check_success.__func__;before=fresh_physical_state(n)
    result=bool(n.check_success());attempt=dict(native_success=result,planner_success=True)
    calls=[];old_profile=sys.getprofile()
    def profile(frame,event,arg):
        if event=='call' and frame.f_code is original.__code__:calls.append(event)
    try:
        sys.setprofile(profile)
        state,receipt=terminal.observe(n,attempt,frozen_policy=terminal.policy(terminal.TASK))
    finally:sys.setprofile(old_profile)
    assert calls==[] and state==before==fresh_physical_state(n)
    assert n.check_success.__func__ is original and bool(n.check_success())==result
    assert receipt['component_result']['conjunction'] is not failure
    if failure:assert receipt['component_result']['components']['left_xy']['passed'] is False


def test_native_strict_boundaries_and_quaternion_sign_preserved(monkeypatch):
    n,_=environment(monkeypatch);count=0
    for dtype in (np.float32,np.float64):
        for offset in (.04999999,.05,.05000001):
            for dz in (.02999999,.03,.03000001):
                for sign in (-1.,1.):
                    n.left_shoe.pose.p=np.array([offset,-.17,.65+dz],dtype=dtype)
                    n.left_shoe.pose.q=np.array([.5,.5,-.5,-.5],dtype=dtype)*sign
                    n.shoe_box.pose.p=np.array([0.,-.13,.64],dtype=dtype)
                    attempt=dict(native_success=bool(n.check_success()),planner_success=True)
                    terminal.observe(n,attempt,frozen_policy=terminal.policy(terminal.TASK));count+=1
    assert count==36


@pytest.mark.parametrize('change',['source','predicate_replaced','role','duplicate','state_mutation','result','policy'])
def test_invalid_observation_cannot_claim_verified_components(monkeypatch,tmp_path,change):
    n,module=environment(monkeypatch);attempt=dict(native_success=True,planner_success=True);policy=terminal.policy(terminal.TASK)
    if change=='source':
        path=tmp_path/'wrong.py';path.write_text('# drift\n');module.__file__=str(path)
    elif change=='predicate_replaced':n.check_success=lambda:True
    elif change=='role':n.left_shoe.name='not_a_shoe'
    elif change=='duplicate':n.right_shoe=n.left_shoe
    elif change=='state_mutation':
        getter=n.left_shoe.get_pose
        def changed():
            n.left_shoe.pose.p[0]+=.001;return getter()
        n.left_shoe.get_pose=changed
    elif change=='result':attempt['native_success']=False
    else:policy['numpy_version']='3.0'
    with pytest.raises(ValueError):terminal.observe(n,attempt,frozen_policy=policy)


def failed_output(tmp_path,monkeypatch):
    n,_=environment(monkeypatch);n.left_shoe.pose.p[0]=.06
    attempt=dict(task=terminal.TASK,seed=10000,native_success=False,planner_success=True,initial_scene={},terminal_observer_invoked=True)
    policy=terminal.policy(terminal.TASK)
    timing.terminal_observer(tmp_path,policy)(n,attempt)
    timing.atomic_json(tmp_path/'expert_attempt.json',attempt)
    timing.atomic_json(tmp_path/'diagnostic_manifest.json',dict(kind=timing.KIND,version=3,task=terminal.TASK,
        seed=10000,design_sha256='d'*64,expert_terminal_policy=policy,**timing.FLAGS))
    timing.atomic_json(tmp_path/'run.json',dict(kind=timing.KIND,task=terminal.TASK,seed=10000,
        status='blocked_capture',native_episodes_started=1,**timing.FLAGS))
    timing.commit_index(tmp_path)
    assert timing.verify_output(tmp_path,'d'*64)['status']=='blocked_capture'


@pytest.mark.parametrize('tamper',['component','role','physical','native_result','missing','policy','downgrade','invocation'])
def test_failed_evidence_recomputed_after_index_rehash(tmp_path,monkeypatch,tamper):
    failed_output(tmp_path,monkeypatch)
    name='expert_terminal_evidence.json';value=timing.read(tmp_path/name)
    if tamper=='component':value['component_result']['components']['left_xy']['passed']=True
    elif tamper=='role':value['named_roles']['left_shoe']['p'][0]=0
    elif tamper=='physical':
        name='expert_final_physical_state.json';value=timing.read(tmp_path/name);value['actors'][0]['p'][0]=.2
    elif tamper=='native_result':value['original_native_success']=True
    elif tamper=='policy':value['policy']['native_source_sha256'][terminal.TASK]='a'*64
    elif tamper=='downgrade':
        name='diagnostic_manifest.json';value=timing.read(tmp_path/name);value['version']=2
    elif tamper=='invocation':
        name='expert_attempt.json';value=timing.read(tmp_path/name);value.pop('terminal_observer_invoked')
    else:(tmp_path/name).unlink()
    if tamper!='missing':timing.atomic_json(tmp_path/name,value)
    timing.commit_index(tmp_path)
    with pytest.raises(ValueError):timing.verify_output(tmp_path,'d'*64)


@pytest.mark.parametrize('original_error',[None,'expert_exception','native_false'])
@pytest.mark.parametrize('observer_error',[False,True])
def test_optional_hook_after_original_result_before_cleanup_preserves_failure(tmp_path,monkeypatch,original_error,observer_error):
    from scripts import collect_robotwin_clean50
    n,_=environment(monkeypatch);order=[]
    if original_error=='native_false':n.left_shoe.pose.p[0]=.06
    def expert(env):
        order.append('expert');env._take_picture()
        if original_error=='expert_exception':raise RuntimeError('original expert error')
    monkeypatch.setattr(collect_robotwin_clean50,'play_expert',expert)
    monkeypatch.setattr(capture,'install_base_frame_adapter',lambda *a,**k:{})
    row={arm:dict(pose=[0.,0.,.8,1.,0.,0.,0.],gripper=1.) for arm in ('left','right')}
    monkeypatch.setattr(capture,'sample',lambda native:deepcopy(row))
    monkeypatch.setattr(capture,'reference_bundle',lambda *a:({},{}))
    n.close_env=lambda:order.append('close')
    def hook(native,attempt):
        assert native is n and 'close' not in order;order.append('observer')
        assert attempt['native_success']==(None if original_error=='expert_exception' else original_error!='native_false')
        if observer_error:raise OSError('observer failed')
    path=tmp_path/'attempt.json'
    call=lambda:capture.capture_reference(lambda seed:n,{'task':terminal.TASK},10000,set(),attempt_path=path,terminal_observer=hook)
    if original_error:
        with pytest.raises((RuntimeError,ValueError),match='original expert error|Reference native expert'):call()
    else:call()
    audit=timing.read(path)
    assert order==['expert','observer','close']
    assert ('terminal_observation_error' in audit) is observer_error
    if original_error=='expert_exception':assert audit['error']=='RuntimeError: original expert error'
    if original_error=='native_false':assert audit['native_success'] is False


def test_required_terminal_observation_failure_retains_state_and_blocks_replay(tmp_path,monkeypatch):
    from scripts import collect_robotwin_clean50
    n,_=environment(monkeypatch);factories=[]
    monkeypatch.setattr(collect_robotwin_clean50,'play_expert',lambda native:native._take_picture())
    monkeypatch.setattr(capture,'install_base_frame_adapter',lambda *a,**k:{})
    row={arm:dict(pose=[0.,0.,.8,1.,0.,0.,0.],gripper=1.) for arm in ('left','right')}
    monkeypatch.setattr(capture,'sample',lambda native:deepcopy(row))
    monkeypatch.setattr(capture,'reference_bundle',lambda *a:({},dict(signature=[])))
    class Observer:
        def __init__(self,*a,**k):pass
        def start(self):pass
        def stop(self,**k):pass
        def receipt(self):return {'timing_compatibility_ready':True}
    monkeypatch.setattr(timing,'ControlTimingObserver',Observer)
    monkeypatch.setattr(timing,'replay',lambda *a:pytest.fail('Failed terminal evidence must not launch replay'))
    def failure(*a,**k):raise ValueError('component audit failed')
    monkeypatch.setattr(terminal,'observe',failure)
    def factory(seed):factories.append(seed);return n
    result=timing.run_task(tmp_path,terminal.TASK,[dict(task=terminal.TASK)],
        dict(excluded_training_scene_sha256=[]),[],factory,terminal_policy=terminal.policy(terminal.TASK))
    assert result['status']=='blocked_capture' and factories==[10000] and n.closed
    assert (tmp_path/'expert_final_physical_state.json').exists()
    assert not (tmp_path/'legacy_outcome.json').exists()
    assert timing.read(tmp_path/'expert_attempt.json')['native_success'] is True
    assert timing.read(tmp_path/'expert_attempt.json')['terminal_observation_error']=='ValueError: component audit failed'
