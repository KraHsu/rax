import ast
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import transforms3d as t3d

from services.robotwin import cabinet_terminal_evidence as terminal
from services.robotwin import expert_planner_observer as planner
from services.robotwin.success_metadata import MARKER, fresh_physical_state, initialize_success_metadata
from services.robotwin.collection import initial_scene,scene_fingerprint
from test_robotwin_success_metadata import Entity, environment as old_environment,load_node
from test_robotwin_expert_terminal_evidence import native_class

ROOT=Path(__file__).parents[1]/'fixtures'
S=ROOT/'robotwin_terminal_evidence';M=ROOT/'robotwin_success_metadata';F=ROOT/'robotwin_failure_observation'

class Pose:
    def __init__(self,p,q):self.p=np.asarray(p,np.float32);self.q=np.asarray(q,np.float32)
    def to_transformation_matrix(self):
        x=np.eye(4,dtype=np.float32);x[:3,:3]=t3d.quaternions.quat2mat(self.q);x[:3,3]=self.p;return x


def terminal_native(monkeypatch):
    n,_=old_environment(terminal.TASK,monkeypatch)
    Actor,_=native_class(F/'actor_utils.py','Actor',monkeypatch,
        methods={'get_pose','get_name','get_functional_point'},sapien=SimpleNamespace(Pose=Pose),Literal=Literal,np=np)
    Articulation,_=native_class(F/'actor_utils.py','ArticulationActor',monkeypatch,
        methods={'get_point','get_qpos','get_qvel'},Actor=Actor,sapien=SimpleNamespace(Pose=Pose),Literal=Literal,np=np,t3d=t3d)
    node=next(n for n in ast.parse((F/'actor_utils.py').read_text()).body if getattr(n,'name',None)=='ArticulationActor')
    Articulation.POINTS=ast.literal_eval(next(n.value for n in node.body if isinstance(n,ast.Assign) and n.targets[0].id=='POINTS'))
    # native_class strips bases when selecting methods; restore only audited inherited readers.
    for name in ('get_pose','get_name','get_functional_point'):setattr(Articulation,name,getattr(Actor,name))
    Robot,_=native_class(S/'robot.py','Robot',monkeypatch,
        methods={'is_left_gripper_open','is_right_gripper_open','get_left_gripper_val','get_right_gripper_val'})
    oldrobot=n.robot;n.robot=Robot();n.robot.__dict__.update(oldrobot.__dict__)
    for name in ('is_left_gripper_open','is_right_gripper_open','get_left_gripper_val','get_right_gripper_val'):
        n.robot.__dict__.pop(name,None)
    n.robot.left_gripper=n.robot.right_gripper=[object()];n.robot.left_gripper_val=n.robot.right_gripper_val=1.
    obj=n.object;cab=n.cabinet;obj.pose=Pose(obj.pose.p,obj.pose.q);cab.pose=Pose(cab.pose.p,cab.pose.q)
    n.object=Actor.__new__(Actor);n.object.actor=obj
    n.cabinet=Articulation.__new__(Articulation);n.cabinet.actor=cab
    cfg=json.loads((F/'cabinet_model_data.json').read_text());n.cabinet.config=deepcopy(cfg)
    link=Entity('link_1');link.pose=Pose([.25,.2,.8],[1,0,0,0]);cab.get_links=lambda:[link];n.cabinet.link_dict={'link_1':link}
    initialize_success_metadata(terminal.TASK,n,expected_scene_sha256=scene_fingerprint(initial_scene(n)))
    return n,obj,link


def test_cabinet_terminal_keeps_original_skipped_predicate_none(monkeypatch):
    n,obj,link=terminal_native(monkeypatch);n.plan_success=False
    before=fresh_physical_state(n);attempt={'scene_sha256':getattr(n,MARKER)['scene_before'],'success_metadata':getattr(n,MARKER),'native_success':None,'planner_success':False}
    state,r=terminal.observe(n,attempt,frozen_policy=terminal.policy(F/'cabinet_model_data.json'))
    assert {k:v for k,v in state.items()if k!='observed_articulation_links'}==before==fresh_physical_state(n)
    assert r['original_native_success']is None and not r['original_predicate_evaluated']
    assert r['component_result']['conjunction']==bool(n.check_success())
    assert all(r[k]==0 for k in ('extra_physics_steps','extra_control_calls','extra_get_info_calls','extra_predicate_calls'))


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('rise',[.00699999,.007,.00700001,.11999999,.12,.12000001])
def test_cabinet_original_AST_strict_numeric_boundaries(monkeypatch,dtype,rise):
    n,obj,link=terminal_native(monkeypatch);n.origin_z=dtype(n.origin_z)
    # Rebuild the fixture from a same-dtype initial state to retain exact provenance.
    obj.pose.p=obj.pose.p.astype(dtype);n.origin_z=None;delattr(n,MARKER);del n.origin_z;del n.arm_tag
    initialize_success_metadata(terminal.TASK,n,expected_scene_sha256=scene_fingerprint(initial_scene(n)))
    target=n.cabinet.get_functional_point(0);obj.pose.p=np.asarray([target[0],target[1],float(n.origin_z)+rise],dtype=dtype)
    attempt={'scene_sha256':getattr(n,MARKER)['scene_before'],'success_metadata':getattr(n,MARKER),'native_success':bool(n.check_success()),'planner_success':True}
    _,r=terminal.observe(n,attempt,frozen_policy=terminal.policy(F/'cabinet_model_data.json'))
    assert r['component_result']['conjunction']==attempt['native_success']


@pytest.mark.parametrize('change',['asset','origin','arm','state','target','label'])
def test_terminal_derived_evidence_rehash_cannot_hide_operand_change(monkeypatch,change):
    n,_,_=terminal_native(monkeypatch);attempt={'scene_sha256':getattr(n,MARKER)['scene_before'],'success_metadata':getattr(n,MARKER),'native_success':None,'planner_success':False};p=terminal.policy(F/'cabinet_model_data.json');state,r=terminal.observe(n,attempt,frozen_policy=p)
    if change=='asset':p['asset_metadata']['functional_points'][0]['matrix'][0][3]+=.1
    elif change=='origin':r['origin_z']['value']+=.1
    elif change=='arm':r['arm_tag']='left'
    elif change=='state':r['object']['p'][0]+=.1
    elif change=='target':r['target_xyz'][0]+=.1
    else:r['original_native_success']=False
    with pytest.raises(ValueError):terminal.validate_receipt(r,state,frozen_policy=p,attempt=attempt)


def planner_native(monkeypatch):
    ArmTag,_=load_node(M/'action.py','ArmTag',monkeypatch)
    names=set(planner.SCOPES)
    Base,_=native_class(S/'_base_task.py','Base_Task',monkeypatch,methods=names,
        ArmTag=ArmTag,Action=object,Actor=object,Literal=Literal,np=np,sapien=SimpleNamespace(Pose=Pose),deepcopy=deepcopy)
    Robot,_=native_class(S/'robot.py','Robot',monkeypatch,methods=set(planner.REQUESTS),np=np,deepcopy=deepcopy)
    Native,_=native_class(M/'put_object_cabinet.py',terminal.TASK,monkeypatch,Base_Task=Base,ArmTag=ArmTag,np=np)
    n=Native();n.plan_success=True;n.need_plan=True;n.left_joint_path=[];n.right_joint_path=[];n.robot=Robot();n.robot.communication_flag=False
    n.robot.left_entity=Entity('left');n.robot.right_entity=Entity('right');n.robot._trans_from_end_link_to_gripper=lambda p,arm_tag:deepcopy(p)
    n.robot.get_constraint_pose=lambda x,arm_tag:deepcopy(x)
    return n


def test_real_source_control_arguments_and_return_objects_unchanged(monkeypatch):
    n=planner_native(monkeypatch);calls=[];output={'status':'Success','position':np.ones((2,3)),'velocity':np.zeros((2,3))}
    def plan(*args,**kwargs):calls.append((deepcopy(args),deepcopy(kwargs)));return output
    n.robot.left_planner=SimpleNamespace(plan_path=plan);p=[0.,0.,.8,1.,0.,0.,0.];c=[1,1,1,0,0,0]
    baseline=n.left_move_to_pose(p,constraint_pose=c);baseline_call=calls[-1]
    obs=planner.ExpertPlannerObserver(n,frozen_policy=planner.policy()).start()
    try:observed=n.left_move_to_pose(p,constraint_pose=c)
    finally:obs.stop()
    assert baseline is observed is output
    for x,y in zip(calls[-1][0],baseline_call[0],strict=True):assert np.array_equal(x,y)
    assert calls[-1][1]==baseline_call[1] and p==[0.,0.,.8,1.,0.,0.,0.]
    receipt=obs.receipt();assert not receipt['errors'] and receipt['first_true_to_false_scope']is None
    assert len([e for e in receipt['events']if e['event']=='call'and e['method']=='left_plan_path'])==1
    assert sys.getprofile()is None
    with pytest.raises(ValueError):obs.start()


def test_first_failed_request_separate_from_first_fatal_native_scope(monkeypatch):
    from scripts.observe_robotwin_expert_failure import validate_planner_receipt
    n=planner_native(monkeypatch);n.robot.left_planner=SimpleNamespace(plan_path=lambda *a,**k:{'status':'Failure'})
    obs=planner.ExpertPlannerObserver(n,frozen_policy=planner.policy()).start()
    try:
        # A directly observed request can fail without modifying native.plan_success.
        n.robot.left_plan_path([0,0,.8,1,0,0,0]);assert n.plan_success
        n.left_move_to_pose([0,0,.8,1,0,0,0]);assert n.plan_success is False
    finally:obs.stop()
    r=obs.receipt();assert r['first_non_success_request_id']==1
    assert r['first_true_to_false_scope']['method']=='left_move_to_pose'
    assert r['first_true_to_false_scope']['call_id']!=r['first_non_success_request_id']
    assert r['first_true_to_false_scope']['last_recognized_non_success_request']==3
    validate_planner_receipt(r,planner.policy())
    changed=deepcopy(r)
    changed['first_true_to_false_scope']['last_recognized_non_success_request']=1
    with pytest.raises(ValueError,match='Fatal scope attribution'):
        validate_planner_receipt(changed,planner.policy())


def test_exception_unwind_preserves_original_exception_and_is_not_failure_status(monkeypatch):
    n=planner_native(monkeypatch)
    def error(*a,**k):raise RuntimeError('original planning error')
    n.robot.left_planner=SimpleNamespace(plan_path=error)
    obs=planner.ExpertPlannerObserver(n,frozen_policy=planner.policy()).start()
    try:
        with pytest.raises(RuntimeError,match='original planning error'):n.left_move_to_pose([0,0,.8,1,0,0,0])
    finally:obs.stop()
    assert obs.receipt()['first_non_success_request_id']is None and not obs.receipt()['errors']


def test_cpu_serializer_never_touches_unrecognized_device_object():
    class Device:
        def __repr__(self):raise AssertionError('repr')
        def cpu(self):raise AssertionError('GPU transfer')
        def __array__(self,*a):raise AssertionError('coercion')
    assert planner.cpu_value(Device())['observed']is False
