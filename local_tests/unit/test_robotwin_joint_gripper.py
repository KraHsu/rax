"""Cross-arm acceptance is not a product of independent arm predicates."""
import copy
import json
import numpy as np
import pytest
from services.robotwin.constraint_schema import HDF5_KEY, validate_bundle, validate_constraint
from services.robotwin.programmatic_annotation import instantiate, OutsideTemplate, validate_template
from services.robotwin.annotation import reference_values
from services.robotwin.constraint_coverage import coverage_report
from services.robotwin import constraint_refinement as gate
from test_robotwin_constraint_refinement import row
from test_robotwin_acceptance import template as base_template


def joint_rule():
    return dict(id='joint_pre', type='gripper_pair_max_interval', subjects='both_grippers',
                selector=dict(kind='joint_gripper_plateaus',state='before_joint_closed',after='episode_start'),
                origin='world',orientation='world',parameters={},rationale='Joint reference plateau, candidate only',
                allowed_margin=dict(lower=.25,upper=1.,unit='fraction'))


def fixture(left=(.5,.5,.5,0,0,0),right=(.5,.5,.5,0,0,0)):
    class Handle(dict): pass
    h=Handle(); length=len(left)
    bundle=dict(schema_version='1.4',action_contract=dict(position='absolute',rotation='absolute',
        quaternion_order='wxyz',gripper='absolute_open_fraction',frame='world'),
        provenance=dict(split='train',variant='clean_50',recording_id='test',source='test',native_success=True),constraints=[])
    h[HDF5_KEY]=np.asarray(json.dumps(bundle));h.attrs={'robotwin_expert_provenance':json.dumps({'task':'adjust_bottle'})}
    for arm,grip in [('left',left),('right',right)]:
        h[f'endpose/{arm}_endpose']=np.tile([0,0,0,1,0,0,0],(length,1))
        h[f'endpose/{arm}_gripper']=np.asarray(grip)
    old=dict(id='all',type='gripper_interval',subjects='active_arms',selector=dict(kind='gripper_plateaus',state='all',after='episode_start'),
        origin='world',orientation='world',parameters={},rationale='Original all plateau',allowed_margin=dict(lower=.25,upper=.25,unit='fraction'))
    t=base_template();t['signatures']=[['','']];t['rules']=[old,dict(old,id='closed',selector=dict(kind='gripper_plateaus',state='closed',after='episode_start')),
        dict(id='geometry',type='point_distance_interval',subjects='active_arms',selector=dict(kind='terminal'),
             origin='selection_start',orientation='world',parameters={},rationale='Position')]
    return h,bundle,t


def migrate(t):
    new=copy.deepcopy(t);new['rules'][0]['reference_phase']='after_joint_closed';new['rules'].append(joint_rule());return new


def test_reassembly_reference_mask_and_preserved_post_closed_rules():
    h,b,t=fixture();old,_=instantiate(h,t);cs,audit=instantiate(h,migrate(t))
    pair=next(c for c in cs if c['type']=='gripper_pair_max_interval')
    assert pair['parameters']['bounds']==[.25,1.] and pair['penalty']['scale']==.02
    assert pair['activation']==dict(mode='interval',start=0,end=3)
    for c in cs:
        if c['id'].startswith(('all_','closed_')):
            prior=next(x for x in old if x['id']==c['id'])
            assert {k:v for k,v in c.items() if k!='evidence'}=={k:v for k,v in prior.items() if k!='evidence'}
    assert len(audit['joint_gripper_reference']['replaced_pre_frames'])==4
    for left,right,passes in [(0,.5,True),(.5,0,True),(0,0,False)]:
        actual=copy.deepcopy(h)
        actual['endpose/left_gripper'][:3]=left; actual['endpose/right_gripper'][:3]=right
        values=reference_values(actual,pair)
        assert bool((values>=.25).all())==passes
    report=coverage_report(cs,6);assert report['complete']
    assert all(report['per_arm'][a]['gripper']['covered']==5 for a in ('left','right'))
    b['constraints']=cs;validate_bundle(b,6)


def test_asynchronous_other_arm_ramp_keeps_max_plateau_and_transfer_coverage():
    h,_,t=fixture(left=(.5,.25,0,0,0,0),right=(.5,.5,.5,0,0,0))
    _,audit=instantiate(h,migrate(t))
    assert audit['joint_gripper_reference']['windows']==[dict(mode='interval',start=0,end=3)]
    assert audit['joint_gripper_reference']['first_joint_closed_stable']==3


def test_no_boundary_or_uncovered_transferred_platform_fails_closed():
    h,_,t=fixture(right=(.5,.5,.5,.5,.5,.5))
    t['signatures']=[['']]
    with pytest.raises(OutsideTemplate,match='boundary'):instantiate(h,migrate(t))
    h,_,t=fixture(left=(.4,.4,.4,0,0,0),right=(.8,.7,.6,0,0,0))
    t['signatures']=[['','close']]
    with pytest.raises(OutsideTemplate,match='uncovered|Missing selection'):instantiate(h,migrate(t))


def test_schema_rejects_pair_as_single_arm_or_event_and_old_version():
    h,b,t=fixture();cs,_=instantiate(h,migrate(t));c=next(c for c in cs if c['type']=='gripper_pair_max_interval')
    for change in [('subject','right_eef'),('reference_frame',dict(origin_m=[1,0,0],quaternion_wxyz=[1,0,0,0])),('activation',dict(mode='endpoint',start=0,end=3))]:
        bad=copy.deepcopy(c);bad[change[0]]=change[1]
        with pytest.raises(ValueError):validate_constraint(bad)
    b['constraints']=[c];b['schema_version']='1.3'
    with pytest.raises(ValueError,match='1.4'):validate_bundle(b)
    t=migrate(t);t['rules'][-1]['selector']['state']='closed'
    with pytest.raises(ValueError):validate_template(t)


def test_migration_is_only_explicit_evidence_bound_data_edit():
    _,_,parent=fixture();dev=gate.development_summary([row()])
    patch=dict(version=1,base_template_sha256=gate.template_hash(parent),reason='Joint recombination evidence',
               evidence_ids=[dev['rows'][0]['trial_id']],operations=[dict(op='replace_pregrasp_gripper_with_pair',rule_ids=['all'],joint_rule=joint_rule())])
    proposal=gate.propose_revision(parent,patch,dev);assert proposal['state']=='revision_proposed'
    assert proposal['template']==migrate(parent)
    old=copy.deepcopy(parent['rules'][0]);old['reference_phase']='after_joint_closed'
    patch['operations']=[dict(op='replace_rule',rule=old)]
    with pytest.raises(ValueError,match='activation'):gate.propose_revision(parent,patch,dev)
    patch['operations']=[dict(op='replace_pregrasp_gripper_with_pair',rule_ids=['closed'],joint_rule=joint_rule())]
    with pytest.raises(ValueError,match='all-gripper'):gate.propose_revision(parent,patch,dev)


def test_pair_embedded_hdf5_and_viewer_curve(tmp_path):
    import h5py
    from services.dataset_viewer.store import feature_curve
    from services.robotwin.constraint_schema import read_embedded
    h,b,t=fixture();cs,_=instantiate(h,migrate(t));b['constraints']=cs
    with h5py.File(tmp_path/'pair.hdf5','w') as f:
        for k,v in h.items(): f.create_dataset(k,data=json.dumps(b) if k==HDF5_KEY else v)
        pair=next(c for c in read_embedded(f)['constraints'] if c['type']=='gripper_pair_max_interval')
        np.testing.assert_equal(feature_curve(f,pair),[.5,.5,.5,0,0,0])
        np.testing.assert_equal(reference_values(f,pair),[.5,.5,.5])


def test_native_acceptance_scores_joint_feature_without_right_arm_fallback():
    from services.robotwin.acceptance import trajectory_violations
    h,b,t=fixture();cs,_=instantiate(h,migrate(t));b['constraints']=cs
    b['provenance'].update(split='validation',variant='clean_heldout',source='acceptance_reference_not_for_training')
    ref={arm:dict(pose=h[f'endpose/{arm}_endpose'].tolist(),gripper=h[f'endpose/{arm}_gripper'].tolist()) for arm in ('left','right')}
    for grips,accepted in [((0,.5),True),((.5,0),True),((0,0),False)]:
        actual=copy.deepcopy(ref)
        for arm,g in zip(('left','right'),grips):actual[arm]['gripper'][1:3]=[g,g]
        result=trajectory_violations(b,ref,actual)
        assert result['passed']==accepted


def test_episode_recorder_embeds_joint_feature_with_full_motion_coverage(tmp_path):
    import h5py
    from services.robotwin.recording import EpisodeRecorder
    from services.robotwin.constraint_schema import read_embedded
    from test_robotwin_constraints import observation
    h,_,t=fixture();cs,_=instantiate(h,migrate(t));pair=next(c for c in cs if c['type']=='gripper_pair_max_interval')
    path=tmp_path/'data/episode0.hdf5';recorder=EpisodeRecorder(path,recording_id='joint-test',instruction='test')
    recorder.upsert(pair,0)
    for i in range(6):
        obs=observation(0)
        for a in ('left','right'):
            obs['endpose'][a+'_endpose']=h[f'endpose/{a}_endpose'][i]
            obs['endpose'][a+'_gripper']=h[f'endpose/{a}_gripper'][i]
        recorder.append(obs)
    recorder.finish(success=True)
    with h5py.File(path) as f:
        bundle=read_embedded(f)
        assert bundle['schema_version']=='1.4' and pair in bundle['constraints']
        assert coverage_report(bundle['constraints'],6)['complete']
