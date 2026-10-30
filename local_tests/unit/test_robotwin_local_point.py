"""Rigid-marker geometry, compilation and portable dataset semantics."""
import copy
import json
import math

import h5py
import numpy as np
import pytest

from services.robotwin.annotation import constraint_positions, reference_values
from services.robotwin.constraint_schema import HDF5_KEY, SCHEMA_VERSION, canonical_json, read_embedded, validate_bundle, validate_constraint
from services.robotwin.programmatic_annotation import instantiate, validate_template
from services.dataset_viewer.store import feature_curve
from test_robotwin_joint_gripper import fixture


def local_point(offset=(.2, 0., 0.)):
    return dict(frame='subject_eef', offset_m=list(offset), provenance=dict(source='known_geometry',
        reference='synthetic rigid marker', sha256='a'*64, rationale='Analytic test geometry, not a contact point'))


def constraint(kind='axis_position_interval', *, anchor='predicted_start'):
    p = dict(bounds=[0., 0.], local_point=local_point())
    if kind != 'point_distance_interval': p['axis'] = [1., 0., 0.]
    if kind == 'axis_displacement_interval': p['anchor'] = anchor
    return dict(id='marker',type=kind,subject='left_eef',parameters=p,
        reference_frame=dict(origin_m=[1.,2.,0.],quaternion_wxyz=[math.sqrt(.5),0,0,math.sqrt(.5)]),
        activation=dict(mode='interval',start=0,end=3),penalty=dict(kind='squared_hinge',scale=.1,weight=1),
        evidence=dict(source='known_geometry',status='accepted',report_id='synthetic',rationale='Analytic geometry only'))


def handle():
    h,b,t = fixture(left=(1.,1.,1.),right=(1.,1.,1.))
    h['endpose/left_endpose'] = np.array([[1,2,0,1,0,0,0], [1,2,0,math.sqrt(.5),0,0,math.sqrt(.5)], [1,2,0,0,0,0,1]])
    return h,b,t


@pytest.mark.parametrize('kind,expected', [
    ('axis_position_interval',[0.,.2,0.]),
    ('radial_distance_interval',[.2,0.,.2]),
    ('point_distance_interval',[.2,.2,.2]),
    ('axis_displacement_interval',[0.,.2,0.]),
])
def test_analytic_rotated_point_and_reference_basis(kind,expected):
    h,_,_=handle();c=constraint(kind)
    np.testing.assert_allclose(reference_values(h,c),expected,atol=1e-15)
    np.testing.assert_allclose(feature_curve(h,c),expected,atol=1e-15)
    if kind=='axis_displacement_interval':
        c['parameters']['axis']=[0.,1.,0.]  # Reference y is negative world x.
        np.testing.assert_allclose(reference_values(h,c),[0,.2,.4],atol=1e-15)


def test_zero_offset_retains_position_only_even_without_valid_orientation():
    h,_,_=handle();pose=h['endpose/left_endpose'];pose[:,3:]=np.nan
    np.testing.assert_array_equal(constraint_positions(pose,{}),pose[:,:3])
    np.testing.assert_array_equal(constraint_positions(pose,{'local_point':local_point((0,0,0))}),pose[:,:3])
    with pytest.raises(ValueError,match='quaternion'):constraint_positions(pose,{'local_point':local_point()})


@pytest.mark.parametrize('scale',[1.+5e-8,1.+8e-5])
def test_marker_normalizes_valid_measured_quaternion_roundoff_only(scale):
    h,_,_=handle();pose=h['endpose/left_endpose'].copy()
    expected=constraint_positions(pose,{'local_point':local_point()})
    pose[:,3:]*=scale
    np.testing.assert_allclose(constraint_positions(pose,{'local_point':local_point()}),expected,atol=1e-15)
    pose[:,3:]*=1.01
    with pytest.raises(ValueError,match='quaternion'):constraint_positions(pose,{'local_point':local_point()})


@pytest.mark.parametrize('change', ['frame','units','source','hash','missing_provenance','nonfinite','shape','extra'])
def test_schema_rejects_ambiguous_or_unproven_marker(change):
    c=constraint();p=c['parameters']['local_point']
    if change=='frame':p['frame']='world'
    elif change=='units':p['offset_mm']=p.pop('offset_m')
    elif change=='source':p['provenance']['source']='agent_guess'
    elif change=='hash':p['provenance']['sha256']='A'*64
    elif change=='missing_provenance':del p['provenance']
    elif change=='nonfinite':p['offset_m'][0]=float('nan')
    elif change=='shape':p['offset_m']=[1,2]
    else:p['task']='turn_switch'
    with pytest.raises(ValueError):validate_constraint(c)


def test_schema_version_and_wrong_primitive_fail_closed():
    _,b,_=handle();b['constraints']=[constraint()]
    with pytest.raises(ValueError,match='1.5'):validate_bundle(b)
    b['schema_version']=SCHEMA_VERSION;validate_bundle(b)
    b['constraints'][0]['type']='axis_angle_interval'
    b['constraints'][0]['parameters']=dict(bounds=[0.,1.],local_axis=[1,0,0],reference_axis=[1,0,0],local_point=local_point())
    with pytest.raises(ValueError):validate_bundle(b)


@pytest.mark.parametrize('origin', ['selection_start','selection_end'])
def test_compiler_uses_marker_origin_and_hdf5_roundtrip(tmp_path,origin):
    h,b,t=handle();t['signatures']=[['']]
    t['rules']=[dict(id='fixed_marker',type='point_distance_interval',subjects='left',
        selector=dict(kind='gripper_plateaus',state='open',after='episode_start'),
        origin=origin,orientation='selection_start',parameters=dict(local_point=local_point()),rationale='Analytic marker')]
    validate_template(t);cs,_=instantiate(h,t)
    c=next(c for c in cs if c['id'].startswith('fixed_marker'))
    expected=[1.2,2.,0.] if origin=='selection_start' else [.8,2.,0.]
    np.testing.assert_allclose(c['reference_frame']['origin_m'],expected,atol=1e-15)
    np.testing.assert_allclose(c['parameters']['bounds'],[0.,.405],atol=1e-15)
    t['rules'][0]['parameters']['local_point']['offset_m'][0]=.8
    assert c['parameters']['local_point']['offset_m'][0]==.2  # No mutable template aliases.
    b.update(schema_version=SCHEMA_VERSION,constraints=cs)
    with h5py.File(tmp_path/'marker.hdf5','w') as f:
        for k,v in h.items():f.create_dataset(k,data=canonical_json(b) if k==HDF5_KEY else v)
        assert read_embedded(f)==json.loads(canonical_json(b))
        np.testing.assert_allclose(feature_curve(f,c),reference_values(f,c))


def test_acceptance_observes_rotation_induced_translation_with_frozen_reference():
    from services.robotwin.acceptance import trajectory_violations
    h,b,_=handle();c=constraint();c['activation']=dict(mode='endpoint',start=0,end=2)
    c['parameters']['bounds']=[.19,.21]
    from services.robotwin.full_trajectory import complete_constraints
    cs,_=complete_constraints(h,[c],'synthetic')
    b.update(schema_version=SCHEMA_VERSION,constraints=cs)
    b['provenance'].update(split='validation',variant='clean_heldout',source='acceptance_reference_not_for_training')
    ref={a:dict(pose=h[f'endpose/{a}_endpose'].tolist(),gripper=h[f'endpose/{a}_gripper'].tolist()) for a in ('left','right')}
    assert trajectory_violations(b,ref,ref)['passed']
    actual=copy.deepcopy(ref);actual['left']['pose'][1][3:]=[1,0,0,0]
    assert not trajectory_violations(b,ref,actual)['passed']
    assert ref['left']['pose'][1][3]==math.sqrt(.5)


def test_long_span_publication_reanchors_at_same_reference_local_point():
    from services.robotwin.full_trajectory import complete_constraints
    h,_,_=handle();n=40
    for arm in ('left','right'):
        h[f'endpose/{arm}_endpose']=np.repeat(h[f'endpose/{arm}_endpose'][1:2],n,axis=0)
        h[f'endpose/{arm}_gripper']=np.ones(n)
    h['endpose/left_endpose'][-1,3:]=[0,0,0,1]
    c=constraint('axis_displacement_interval');c['activation']=dict(mode='endpoint',start=0,end=n)
    c['parameters']['axis']=[0.,1.,0.]
    before=reference_values(h,c)
    cs,_=complete_constraints(h,[c],'synthetic')
    revised=next(x for x in cs if x['id']==c['id'])
    assert revised['parameters']['anchor']=='reference_frame'
    np.testing.assert_allclose(revised['reference_frame']['origin_m'],[1,2.2,0],atol=1e-15)
    np.testing.assert_allclose(reference_values(h,revised),before,atol=1e-15)
    assert c['parameters']['anchor']=='predicted_start'
