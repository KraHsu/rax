import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest

from services.robotwin.tolerance import bracket, features, perturb_program, realization


def test_interventions_freeze_lift_and_change_only_selected_targets():
    move={'arm_tag':'left','action':'move','target_pose':[1,2,3,1,0,0,0], 'target_gripper_pos':None,'args':{}}
    grip={'arm_tag':'left','action':'gripper','target_pose':None,'target_gripper_pos':0.,'args':{}}
    program=[[{'arm':'left','actions':[copy.deepcopy(move),copy.deepcopy(move),grip]}],
             [{'arm':'left','actions':[copy.deepcopy(move)]}], [{'arm':'left','actions':[copy.deepcopy(move)]}]]
    before=copy.deepcopy(program)
    result=perturb_program(program,{'grasp_x':.02,'opening':.1,'terminal_z':-.04})
    assert program==before and result[1]==program[1]
    assert result[0][0]['actions'][1]['target_pose'][0]==1.02
    assert result[0][0]['actions'][2]['target_gripper_pos']==.1
    assert result[2][0]['actions'][0]['target_pose'][2]==2.96


def test_features_use_actual_crossing_and_allow_missing_crossing():
    trace=[{'left_pose':[0,0,z,1,0,0,0],'left_opening':g} for z,g in [(0,1),(.1,.6),(.12,.4),(.3,.0)]]
    f=features(trace,'left');assert f['closure_frame']==2 and np.isclose(f['vertical_displacement_m'],.18)
    for s in trace:s['left_opening']=.8
    assert features(trace,'left')['closure_xyz_m'] is None


def test_bracket_stops_at_first_failure_not_later_success_island():
    results=[{'delta':{'grasp_z':x},'outcome':outcome} for x,outcome in
             [(-.005,'success'),(-.01,'task_failure'),(-.02,'success')]]
    assert bracket(results,'grasp_z',-1)=={'pass':-.005,'nonpass':-.01,'nonpass_kind':'task_failure'}


def test_contact_saturated_offset_is_not_a_valid_geometric_test():
    reference={'grasp_end_xyz_m':[0,0,1]}
    observed={'grasp_end_xyz_m':[0,0,.999]}
    checked=realization(reference,observed,{'grasp_z':-.04})
    assert not checked['realized'] and np.isclose(checked['measured_offset_m']['grasp_z'],-.001)


def test_bracket_rejects_a_boundary_that_fails_on_repeat():
    rows=[{'delta':{'grasp_x':.01},'outcome':s} for s in ('success','task_failure')]
    assert bracket(rows,'grasp_x',1)=={'pass':0.,'nonpass':.01,'nonpass_kind':'inconsistent'}


@pytest.mark.parametrize('delta',[{'openning':.1},{'opening':1.1},{'grasp_x':float('nan')},{'terminal_z':True}])
def test_invalid_interventions_cannot_be_silently_ignored_or_clipped(delta):
    with pytest.raises(ValueError):perturb_program([],delta)


def test_fresh_confirmation_requires_matching_identity_and_initial_geometry(tmp_path):
    from scripts.validate_robotwin_tolerances import Experiment
    source=tmp_path/'source';source.mkdir();output=tmp_path/'output';output.mkdir()
    initial={k:[0.,0.,1.,1.,0.,0.,0.] for k in ('left_pose','right_pose','bottle_functional_pose')}
    for file,value in {'run.json':{'seed':0,'model_id':16},'expert_trace.json':[initial],
                       'expert_program.json':[[{'frozen':True}]],'reference.json':{'example':1}}.items():
        (source/file).write_text(json.dumps(value))
    e=Experiment.__new__(Experiment);e.output=output;e.seed=0;e.native=SimpleNamespace(model_id=16);e.initial=copy.deepcopy(initial)
    e.load_program(source);assert e.program==[[{'frozen':True}]]
    e.seed=1
    with pytest.raises(RuntimeError,match='identity'):e.load_program(source)
    e.seed=0;e.initial['bottle_functional_pose'][0]=.001
    with pytest.raises(RuntimeError,match='geometry'):e.load_program(source)
