"""Source-bound scale role initialization; no Scene, motion or native replay."""
from copy import deepcopy

import numpy as np
import pytest

from services.robotwin.collection import initial_scene, scene_fingerprint
from services.robotwin.instruction_metadata import readonly_instruction_metadata
from services.robotwin.success_metadata import (MARKER, SCALE_IDS, SCALE_OBJECTS, SuccessMetadataError,
    fresh_physical_state, initialize_success_metadata, validate_metadata_receipt)
from test_robotwin_success_metadata import environment, initialize


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('x,expected',[(-.25,'left'),(.25,'right'),(0.,'left'),(-0.,'left'),
    (float(np.nextafter(np.float32(0),np.float32(1))),'right'),
    (float(np.nextafter(np.float32(0),np.float32(-1))),'left')])
def test_exact_initial_x_branch_armtag_type_and_unchanged_physics(monkeypatch,dtype,x,expected):
    native,_=environment('place_object_scale',monkeypatch,x=x)
    native.object.pose.p=native.object.pose.p.astype(dtype)
    before=fresh_physical_state(native);info=deepcopy(native.info);predicate=native.check_success.__func__
    with pytest.raises(AttributeError,match='arm_tag'):native.check_success()
    readonly=readonly_instruction_metadata('place_object_scale',native)
    assert not hasattr(native,'arm_tag')
    receipt=initialize('place_object_scale',native)
    assert str(native.arm_tag)==expected and type(native.arm_tag).__name__=='ArmTag'
    assert receipt['assigned_fields']=={'arm_tag':expected} and not hasattr(native,'origin_z')
    assert receipt['source_inputs']['object_position_dtype']==np.dtype(dtype).name
    assert receipt['info']==readonly['info'] and native.info==info
    assert fresh_physical_state(native)==before and native.check_success.__func__ is predicate
    assert receipt['native_actions_executed'] is False and receipt['native_get_info_called'] is False
    assert initialize('place_object_scale',native)==receipt


@pytest.mark.parametrize('x',[-.25,.25])
def test_native_success_boundaries_and_initial_carrier_gripper_unchanged(monkeypatch,x):
    native,_=environment('place_object_scale',monkeypatch,x=x)
    oracle,_=environment('place_object_scale',monkeypatch,x=x)
    oracle_info=type(oracle).get_info(oracle) # Separate pure AST oracle, never called on the production path.
    receipt=initialize('place_object_scale',native)
    assert receipt['info']==oracle_info
    assert type(native.arm_tag) is type(native).play_once.__globals__['ArmTag']
    assert type(oracle.arm_tag) is type(oracle).play_once.__globals__['ArmTag']
    original_arm=str(native.arm_tag)
    for opening in ((True,False),(False,True),(False,False),(True,True)):
        for env in (native,oracle):
            env.robot.is_left_gripper_open=lambda:opening[0]
            env.robot.is_right_gripper_open=lambda:opening[1]
        for radius in (0.,.034999,.035,.035001,.10):
            for z in (.789999,.79,.790001,.8,1.5):
                for env in (native,oracle):env.object.pose.p=np.array([radius,.1,z],np.float32)
                assert bool(native.check_success())==bool(oracle.check_success())
        for env in (native,oracle):env.object.pose.p=np.array([0.,.1,1.5],np.float32)
        assert bool(native.check_success())==opening[0 if original_arm=='left' else 1]
    # Moving the object across world x=0 does not authorize rebasing the carrier.
    with pytest.raises(SuccessMetadataError,match='rebase'):initialize('place_object_scale',native)
    assert str(native.arm_tag)==original_arm


@pytest.mark.parametrize('object_name',SCALE_OBJECTS)
@pytest.mark.parametrize('scale_id',SCALE_IDS)
def test_all_audited_asset_classes_and_scale_ids_keep_native_tokens(monkeypatch,object_name,scale_id):
    native,_=environment('place_object_scale',monkeypatch)
    native.selected_modelname=object_name;native.object.name=object_name;native.selected_model_id=np.int64(3)
    native.scale_id=np.int64(scale_id)
    receipt=initialize('place_object_scale',native)
    assert receipt['info']=={'{A}':f'072_electronicscale/base{scale_id}','{B}':f'{object_name}/base3','{a}':'right'}
    validate_metadata_receipt('place_object_scale',receipt,receipt['scene_before'])


@pytest.mark.parametrize('change',['coupled_arm','position_dtype','scale_id','object_id_bool','info','duplicate_object','duplicate_scale'])
def test_scale_receipt_rejects_self_consistent_unbound_claims(monkeypatch,change):
    native,_=environment('place_object_scale',monkeypatch);receipt=initialize('place_object_scale',native)
    if change=='coupled_arm':
        receipt['source_inputs']['object_position'][0]*=-1
        receipt['assigned_fields']['arm_tag']='left';receipt['info']['{a}']='left'
    elif change=='position_dtype':receipt['source_inputs']['object_position_dtype']='float64'
    elif change=='scale_id':receipt['source_inputs']['scale_model_id']=99
    elif change=='object_id_bool':receipt['source_inputs']['object_model_id']=True
    elif change=='info':receipt['info']['{a}']='left'
    else:
        name='047_mouse' if change=='duplicate_object' else '072_electronicscale'
        receipt['fresh_state']['actors'].append(deepcopy(next(a for a in receipt['fresh_state']['actors'] if a['name']==name)))
    with pytest.raises(SuccessMetadataError):validate_metadata_receipt('place_object_scale',receipt,receipt['scene_before'])


@pytest.mark.parametrize('change',['object','scale','scale_id','object_id','stale_arm','after_action'])
def test_bad_native_scale_identity_or_late_init_never_partially_assigns(monkeypatch,change):
    native,_=environment('place_object_scale',monkeypatch)
    if change=='object':native.selected_modelname='073_rubikscube';native.object.name='073_rubikscube'
    elif change=='scale':native.scale.name='other'
    elif change=='scale_id':native.scale_id=2
    elif change=='object_id':native.selected_model_id=True
    elif change=='stale_arm':native.arm_tag='left'
    else:native.take_action_cnt=1
    with pytest.raises(SuccessMetadataError):
        initialize_success_metadata('place_object_scale',native,expected_scene_sha256=scene_fingerprint(initial_scene(native)))
    assert not hasattr(native,MARKER)
    if change=='stale_arm':assert native.arm_tag=='left'
    else:assert not hasattr(native,'arm_tag')
