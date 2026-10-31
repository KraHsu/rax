"""Fixed reference tangent identity; no generated mask or guessed world axis."""
import copy
import json
import math

import h5py
import numpy as np
import pytest

from services.robotwin.annotation import reference_values, validate_reference_bounds
from services.robotwin.constraint_schema import (SCHEMA_VERSION, HDF5_KEY, canonical_json,
    make_center_tangent_hint, read_embedded, validate_bundle, validate_constraint)
from services.robotwin.programmatic_annotation import instantiate, validate_template
from test_robotwin_programmatic_annotation import episode, template


def qx(theta):
    return [math.cos(theta/2),math.sin(theta/2),0.,0.]


def angle_constraint(n=100):
    return dict(id='angle_band',type='axis_angle_interval',subject='left_eef',
        reference_frame=dict(origin_m=[0.,0.,0.],quaternion_wxyz=[1.,0.,0.,0.]),
        parameters=dict(bounds=[.5,1.],local_axis=[0.,1.,0.],reference_axis=[0.,1.,0.]),
        activation=dict(mode='interval',start=0,end=n),penalty=dict(kind='squared_hinge',scale=.1,weight=1.),
        evidence=dict(source='demonstration_analysis',status='accepted',report_id='ref0',rationale='Synthetic band'))


def hinted(n=100):
    c=angle_constraint(n);c['parameters']['center_tangent_hint']=make_center_tangent_hint(c,qx(-math.pi/4),'ref0',0)
    return c


def test_fixed_projection_source_hash_and_all_old_schema_compatibility():
    c=hinted();validate_constraint(c,100)
    np.testing.assert_allclose(c['parameters']['center_tangent_hint']['direction'],[0,0,-1],atol=1e-15)
    from test_robotwin_joint_gripper import fixture
    h,b,t=fixture();b['constraints']=[c]
    b['provenance']['recording_id']='ref0'
    for version in ('1.0','1.1','1.2','1.3','1.4','1.5'):
        b['schema_version']=version
        with pytest.raises(ValueError,match='1.6'):validate_bundle(b)
        old=copy.deepcopy(b);old['constraints']=[angle_constraint()];validate_bundle(old)
    b['schema_version']=SCHEMA_VERSION;validate_bundle(b)


@pytest.mark.parametrize('change',['direction','zero','parallel','source_hash','source_frame','source_kind','quaternion',
    'recording','wrong_kind','frame','extra','version_bool'])
def test_unproven_or_invalid_tangent_fails_closed(change):
    c=hinted();hint=c['parameters']['center_tangent_hint']
    if change=='direction':hint['direction']=[1,0,0]
    elif change=='zero':hint['direction']=[0,0,0]
    elif change=='parallel':hint['direction']=[0,1,0]
    elif change=='source_hash':hint['source']['sha256']='a'*64
    elif change=='source_frame':hint['source']['frame']=1
    elif change=='source_kind':hint['source']['kind']='predicted_axis'
    elif change=='quaternion':hint['source']['quaternion_wxyz']=[True,0,0,0]
    elif change=='recording':hint['source']['recording_id']='different'
    elif change=='wrong_kind':c['type']='angular_displacement_interval'
    elif change=='frame':hint['frame']='world'
    elif change=='extra':hint['fallback_axis']=[1,0,0]
    else:hint['version']=True
    with pytest.raises(ValueError):validate_constraint(c)


@pytest.mark.parametrize('theta',[0.,math.pi])
def test_degenerate_reference_does_not_guess_a_world_axis(theta):
    c=angle_constraint()
    with pytest.raises(ValueError,match='nondegenerate'):make_center_tangent_hint(c,qx(theta),'ref0',0)


def test_template_optin_preserves_bounds_masks_numpy_values_and_hdf5_reference_binding(tmp_path):
    path=episode(tmp_path/'ref.hdf5')
    t=template();t['rules']=[dict(id='carry_angle',type='axis_angle_interval',subjects='active_arms',
        selector=dict(kind='gripper_plateaus',state='closed',after='first_close'),origin='world',orientation='world',
        parameters=dict(local_axis=[0,1,0],reference_axis=[0,1,0]),rationale='Synthetic positive-lower angle band')]
    with h5py.File(path,'r+') as h:
        pose=h['endpose/left_endpose'][:];pose[:,3:]=qx(-math.pi/4);h['endpose/left_endpose'][:]=pose
        old,old_audit=instantiate(h,t);t['rules'][0]['angle_center_hint']='selection_start_reference_projection_v1'
        validate_template(t);new,new_audit=instantiate(h,t)
        assert old_audit['coverage']==new_audit['coverage']
        assert len(old)==len(new)
        for before,after in zip(old,new):
            np.testing.assert_array_equal(reference_values(h,before),reference_values(h,after))
            assert before['activation']==after['activation'] and before['penalty']==after['penalty']
            assert before['parameters']['bounds']==after['parameters']['bounds']
        c=next(c for c in new if c['type']=='axis_angle_interval')
        assert c['parameters']['center_tangent_hint']['source']['frame']==2
        b=read_embedded(h);b.update(schema_version=SCHEMA_VERSION,constraints=new)
        h[HDF5_KEY][()]=canonical_json(b)
        assert read_embedded(h)==json.loads(canonical_json(b))
        validate_reference_bounds(b,h)
        # Rehashed, geometrically valid replacement hint still cannot claim a
        # different HDF5 reference orientation; it is never derived at training.
        changed=copy.deepcopy(b);ca=next(c for c in changed['constraints'] if c['type']=='axis_angle_interval')
        ca['parameters']['center_tangent_hint']=make_center_tangent_hint(ca,qx(math.pi/4),'ref0',2)
        validate_bundle(changed,len(pose));h[HDF5_KEY][()]=canonical_json(changed)
        with pytest.raises(ValueError,match='HDF5 reference'):read_embedded(h)


def test_exact_measured_norm_roundoff_source_identity_is_idempotent():
    for scale in (1.,1.+8e-5):
        c=angle_constraint();q=[v*scale for v in qx(.7)]
        c['parameters']['center_tangent_hint']=make_center_tangent_hint(c,q,'ref0',0)
        validate_constraint(c);validate_constraint(copy.deepcopy(c))


def test_old_schema15_local_points_remain_readable():
    from test_robotwin_local_point import handle,constraint
    _,b,_=handle();b.update(schema_version='1.5',constraints=[constraint()]);validate_bundle(b)
