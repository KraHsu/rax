import copy
import json

import h5py
import numpy as np
import pytest

from services.robotwin.collection import atomic_json
from services.robotwin.constraint_schema import HDF5_KEY, read_embedded
from services.robotwin.programmatic_annotation import OutsideTemplate, instantiate, validate_template
from services.robotwin.annotation_templates import install, match, refresh


def rule(name, kind, selector, **kwargs):
    return dict(id=name, type=kind, subjects='active_arms', selector=selector, origin='world',
                orientation='world', parameters={}, rationale='Synthetic reference feature', **kwargs)


def template():
    point=rule('close','point_distance_interval',{'kind':'events','direction':'decreasing','occurrence':'first'})
    point['origin']='selection_start'
    grip=rule('closed','gripper_interval',{'kind':'gripper_plateaus','state':'closed','after':'first_close'})
    height=rule('height','axis_position_interval',{'kind':'terminal'})
    height['parameters']={'axis':[0,0,1]}
    delta=copy.deepcopy(height);delta.update(id='lift',type='axis_displacement_interval',start_anchor='first_close')
    return {'version':1,'task':'adjust_bottle','rationale':'Synthetic task template',
            'signatures':[['close']],'rules':[point,grip,height,delta]}


def episode(path, seed=0, offset=0., arm='left', extra_event=False, fingerprint=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    length=8+seed
    bundle={'schema_version':'1.0','action_contract':{'position':'absolute','rotation':'absolute',
            'quaternion_order':'wxyz','gripper':'absolute_open_fraction','frame':'world'},
            'provenance':{'split':'train','variant':'clean_50','recording_id':f'ref{seed}',
                          'source':'synthetic','native_success':True},'constraints':[]}
    with h5py.File(path,'w') as h:
        h.attrs['robotwin_expert_provenance']=json.dumps({'task':'adjust_bottle','seed':seed,'index':seed,
                                                        'initial_scene_sha256':fingerprint or f'scene{seed}'})
        h.create_dataset(HDF5_KEY,data=json.dumps(bundle),dtype=h5py.string_dtype())
        h.create_dataset('trajectory_constraints/history',data='[]',dtype=h5py.string_dtype())
        h['trajectory_constraints'].attrs['revision']=0
        for subject in ['left','right']:
            pose=np.tile([offset,0.,.8,1.,0.,0.,0.],(length,1));grip=np.ones(length)
            if subject==arm:
                pose[:,2]+=np.linspace(0,.1,length)
                grip[seed+2:]=0.
                if extra_event:grip[-2:]=1.
            h.create_dataset(f'endpose/{subject}_endpose',data=pose)
            h.create_dataset(f'endpose/{subject}_gripper',data=grip)
    return path


def test_template_recomputes_each_pose_event_and_arm_without_fixed_indices(tmp_path):
    outputs=[]
    for seed,offset,arm in [(0,0.,'left'),(3,.25,'right')]:
        with h5py.File(episode(tmp_path/f'{seed}.hdf5',seed,offset,arm)) as h:
            constraints,audit=instantiate(h,template());outputs.append(constraints)
            assert audit['signature']==['close']
            assert all(c['subject']==arm+'_eef' for c in constraints)
            assert constraints[0]['activation']['event']['frame']==seed+2
            assert constraints[0]['reference_frame']['origin_m'][0]==offset
            assert constraints[0]['parameters']['bounds']==[0.,.005]
            assert constraints[1]['activation']['start']==seed+2
            assert constraints[-1]['activation']['start']==seed+2
    assert outputs[0][-1]['parameters']['bounds']!=outputs[1][-1]['parameters']['bounds']


def test_unreviewed_patterns_and_task_mismatch_require_agent(tmp_path):
    with h5py.File(episode(tmp_path/'new.hdf5',extra_event=True)) as h:
        with pytest.raises(OutsideTemplate,match='signature'):instantiate(h,template())
    with h5py.File(episode(tmp_path/'same.hdf5')) as h:
        t=template();t['task']='other'
        with pytest.raises(OutsideTemplate,match='Task mismatch'):instantiate(h,t)


@pytest.mark.parametrize('change',['frame','bounds','axis','interval_anchor','no_displacement_anchor'])
def test_template_rejects_fixed_frames_arbitrary_bounds_and_invalid_geometry(change):
    t=template()
    if change=='frame':t['rules'][0]['selector']['frame']=42
    elif change=='bounds':t['rules'][0]['parameters']['bounds']=[0,10]
    elif change=='axis':t['rules'][2]['parameters']['axis']=[0,0,2]
    elif change=='interval_anchor':
        t['rules'][-1]['selector']={'kind':'gripper_plateaus','state':'closed','after':'first_close'}
    else:del t['rules'][-1]['start_anchor']
    with pytest.raises(ValueError):validate_template(t)


def test_extremum_is_reference_selected_and_orientation_is_radians(tmp_path):
    t=template();t['rules']=t['rules'][:1]
    r=rule('lowest','axis_position_interval',{'kind':'position_extremum','axis':[0,0,1],'which':'min','after':'first_close'})
    r['parameters']={'axis':[0,0,1]};t['rules'].append(r)
    r=rule('angle','axis_angle_interval',{'kind':'terminal'})
    r['parameters']={'local_axis':[1,0,0],'reference_axis':[0,1,0]};t['rules'].append(r)
    with h5py.File(episode(tmp_path/'ep.hdf5')) as h:
        constraints,_=instantiate(h,t)
        assert constraints[1]['activation']=={'mode':'endpoint','start':2,'end':3}
        assert np.allclose(constraints[-1]['parameters']['bounds'],np.pi/2+np.array([-.0872664626,.0872664626]))


def test_registry_requires_three_distinct_seed_and_geometry_references(tmp_path):
    directory=tmp_path/'adjust_bottle/aloha-agilex_clean_50'
    source=episode(directory/'data/episode0.hdf5')
    episode(directory/'raw/episode1.hdf5',seed=1,fingerprint='scene0')
    assert install(tmp_path,template(),source_episode=source,agent_run='fixture')['status']=='candidate'
    assert match(tmp_path,'adjust_bottle',source) is None
    episode(directory/'raw/episode2.hdf5',seed=2)
    episode(directory/'raw/episode3.hdf5',seed=3)
    refresh(tmp_path,'adjust_bottle')
    entry,_,_=match(tmp_path,'adjust_bottle',source)
    assert entry['status']=='active' and {e['seed'] for e in entry['validation']}=={0,2,3}


def test_batch_publishes_constraints_and_leaves_agent_leases_untouched(tmp_path):
    from scripts.run_robotwin_programmatic_annotation import run_once
    from scripts.robotwin_annotation_queue import claim
    directory=tmp_path/'adjust_bottle/aloha-agilex_clean_50'
    atomic_json(tmp_path/'batch_manifest.json',{'tasks':['adjust_bottle'],'total_target':3})
    atomic_json(tmp_path/'annotation_control.json',{'task_templates':True})
    for index in range(3):
        path=episode(directory/'raw'/f'episode{index}.hdf5',seed=index)
        atomic_json(directory/'annotation_queue'/f'episode{index}.json',
                    {'task':'adjust_bottle','index':index,'path':str(path),'annotation_status':'pending'})
    install(tmp_path,template(),source_episode=directory/'raw/episode0.hdf5',agent_run='fixture')
    assert claim(tmp_path,'local')['waiting']
    lease=directory/'annotation_queue/episode2.json';receipt=json.loads(lease.read_text())
    receipt.update(annotation_status='processing',claim_id='agent-existing');atomic_json(lease,receipt)
    results=run_once(tmp_path)
    assert len(results)==2
    with h5py.File(directory/'data/episode1.hdf5') as h:
        assert len(read_embedded(h)['constraints'])==4
        assert json.loads(h['trajectory_constraints/history'][()])[0]['agent_run'].startswith('template:')
    assert json.loads(lease.read_text())['claim_id']=='agent-existing'
    assert (directory/'raw/episode2.hdf5').exists()
