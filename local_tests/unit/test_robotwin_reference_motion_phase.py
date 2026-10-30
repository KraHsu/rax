"""Reference-only phases, compiled endpoints and auditable publication."""
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest

from services.robotwin.annotation import digest, embed_review, reference_values
from services.robotwin.constraint_coverage import chunk_candidate
from services.robotwin.constraint_schema import read_embedded
from services.robotwin.full_trajectory import PREFIX
from services.robotwin.programmatic_annotation import instantiate, legacy_motion_coverage, validate_template
from services.robotwin.reference_motion_phase import InvalidReferenceMotionPhase, select_reference_motion_phase, reference_segment_axis
from test_robotwin_programmatic_annotation import episode, template


def phase_rule(kind='axis_displacement_interval', evaluation='endpoint'):
    return dict(id='phase_feature', type=kind, subjects='active_arms',
        selector=dict(kind='motion_segment', version=1, after='first_close', occurrence='last', evaluation=evaluation),
        origin='selection_start', orientation='world',
        parameters=dict(axis='reference_segment_direction', local_point=dict(frame='subject_eef', offset_m=[.12,0,0],
            provenance=dict(source='manual', reference='Synthetic fixed marker', sha256='f'*64, rationale='Analytic fixture only'))),
        allowed_margin=dict(lower=.001, upper=.001, unit='m'), rationale='A reference motion phase is not a contact phase')


def phase_template(rule=None):
    t=template();t['motion_coverage']=legacy_motion_coverage(t)
    t['rules'].append(phase_rule() if rule is None else rule)
    return t


def phase_episode(path, steps=8, close=2, tail=0):
    n=steps+6+tail;episode(path,seed=n-8)
    with h5py.File(path,'r+') as h:
        h['endpose/left_gripper'][...]=np.r_[np.ones(close),np.zeros(n-close)]
        p=np.tile([0,0,.8,1,0,0,0],(n,1)).astype(float)
        p[6:6+steps,0]=np.arange(1,steps+1)*.004
        p[6+steps:,0]=steps*.004
        h['endpose/left_endpose'][...]=p
    return path


def test_complete_reference_segmentation_does_not_clip_across_boundary():
    p=np.zeros((12,3));p[6:,0]=np.arange(1,7)*.004
    segment,audit=select_reference_motion_phase(p,5)
    assert segment['first_action']==6 and segment['anchor']==5 and segment['end_exclusive']==12
    assert segment['steps']==6 and audit['generated_actions_used'] is False
    with pytest.raises(InvalidReferenceMotionPhase,match='No qualifying') as exc:
        select_reference_motion_phase(p,6)
    assert exc.value.audit['segments'][-1]['anchor']==5  # Never clip the same segment to frame6.


def test_last_qualifying_segment_ignores_single_step_and_tiny_eef_noise():
    p=np.array([[0,0,0],[.004,0,0],[.008,0,0],[.008,0,0],[.008,.00002,0],[.008,.00004,0],[.008,.00004,.001]])
    selected,audit=select_reference_motion_phase(p,0)
    assert selected['anchor']==0 and selected['end_exclusive']==3
    assert len(audit['segments'])==4
    assert not audit['segments'][-1]['qualifies']  # One-step tail.
    assert not audit['segments'][-2]['qualifies']  # Two steps, but net only0.04mm.


def test_cumulative_segment_norm_overflow_is_invalid_before_audit_or_selection():
    p=np.array([[0,0,0],[8e153,0,0],[1.6e154,0,0]])
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(InvalidReferenceMotionPhase,match='Non-finite reference segment') as exc:
            select_reference_motion_phase(p,0)
    assert not exc.value.audit['valid'] and not exc.value.audit['segments']
    json.dumps(exc.value.audit,allow_nan=False)


@pytest.mark.parametrize('steps',[2,32])
def test_predicted_phase_anchor_and_endpoint_are_exact_and_guards_unchanged(tmp_path,steps):
    path=phase_episode(tmp_path/'ep.hdf5',steps=steps,close=5);t=phase_template()
    with h5py.File(path) as h:
        old,_=instantiate(h,template());cs,audit=instantiate(h,t)
        c=next(c for c in cs if c['id'].startswith('phase_feature_'))
        np.testing.assert_allclose(reference_values(h,c),[steps*.004],atol=1e-15)
    assert c['activation']==dict(mode='endpoint',start=5,end=steps+6)
    assert c['parameters']['anchor']=='predicted_start'
    np.testing.assert_allclose(c['parameters']['axis'],[1,0,0],atol=1e-15)
    assert c['penalty']['scale']==.005
    assert [c for c in cs if c['id'].startswith(PREFIX)]==[c for c in old if c['id'].startswith(PREFIX)]
    entry=audit['reference_motion_phases']['selections'][0]
    assert entry['selected']['anchor']==entry['after_frame']==5
    assert entry['derived_axis']['net_distance_m']==pytest.approx(steps*.004)
    assert chunk_candidate(c,5,32)
    assert not chunk_candidate(c,6,32)  # Endpoint in view, original predicted anchor absent.
    assert not chunk_candidate(c,0,steps)  # Endpoint outside this chunk.


def test_33_step_phase_raises_instead_of_switching_to_reference_anchor(tmp_path):
    path=phase_episode(tmp_path/'ep.hdf5',steps=33)
    with h5py.File(path) as h:
        old,old_audit=instantiate(h,template())
        with pytest.raises(InvalidReferenceMotionPhase,match='32-action') as exc:instantiate(h,phase_template())
        after,_=instantiate(h,template())
    assert old==after and old_audit['coverage']['complete']
    assert exc.value.audit['selected']['steps']==33 and exc.value.audit['valid'] is False


def test_selected_tcp_zero_net_is_invalid_and_never_selects_an_earlier_phase():
    p=np.array([[0,0,0],[.005,0,0],[.010,0,0],[.010,0,0],[.010,.005,0],[.010,.010,0]])
    selected,audit=select_reference_motion_phase(p,0)
    assert len([s for s in audit['segments'] if s['qualifies']])==2 and selected['anchor']==3
    physical_point=np.zeros_like(p)  # Can occur through translation/rotation cancellation.
    with pytest.raises(InvalidReferenceMotionPhase,match='near-zero') as exc:
        reference_segment_axis(physical_point,selected,audit)
    assert exc.value.audit['selected']['anchor']==3


def test_physical_tcp_translation_rotation_cancellation_is_rejected(tmp_path):
    path=phase_episode(tmp_path/'ep.hdf5')
    with h5py.File(path,'r+') as h:
        p=h['endpose/left_endpose'][()]
        for i in range(6,len(p)):
            theta=(i-5)*.02;p[i,:3]=[.12-.12*math.cos(theta),-.12*math.sin(theta),.8]
            p[i,3:]=[math.cos(theta/2),0,0,math.sin(theta/2)]
        h['endpose/left_endpose'][...]=p
        with pytest.raises(InvalidReferenceMotionPhase,match='near-zero'):instantiate(h,phase_template())


@pytest.mark.parametrize('change',['optional','unknown_threshold','version','boolean_version','first','after_episode_start','after_last_close','alternate_anchor','bad_axis','wrong_orientation','paired_interval','paired_fixed_axis'])
def test_invalid_or_ambiguous_phase_declarations_fail_closed(change):
    t=phase_template();r=t['rules'][-1]
    if change=='optional':r['required']=False
    elif change=='unknown_threshold':r['selector']['stationary_step_m']=.01
    elif change=='version':r['selector']['version']=2
    elif change=='boolean_version':r['selector']['version']=True
    elif change=='first':r['selector']['occurrence']='first'
    elif change=='after_episode_start':r['selector']['after']='episode_start'
    elif change=='after_last_close':r['selector']['after']='last_close'
    elif change=='alternate_anchor':r['start_anchor']='first_close'
    elif change=='bad_axis':r['parameters']['axis']='generated_direction'
    elif change=='wrong_orientation':r['orientation']='selection_start'
    elif change=='paired_interval':r['selector']['evaluation']='interval'
    else:r['parameters']['axis']=[1,0,0]
    with pytest.raises(ValueError):validate_template(t)


def test_interval_and_fixed_reference_endpoint_are_distinct_from_paired_progress(tmp_path):
    path=phase_episode(tmp_path/'ep.hdf5');paired=phase_template()
    fixed=phase_template(phase_rule('axis_position_interval'))
    region=phase_rule('axis_position_interval','interval');region['parameters']['axis']=[0,0,1]
    with h5py.File(path) as h:
        pc,_=instantiate(h,paired);fc,_=instantiate(h,fixed);rc,_=instantiate(h,phase_template(region))
    p=next(c for c in pc if c['id'].startswith('phase_feature'))
    f=next(c for c in fc if c['id'].startswith('phase_feature'))
    r=next(c for c in rc if c['id'].startswith('phase_feature'))
    assert p['parameters']['anchor']=='predicted_start' and 'anchor' not in f['parameters']
    assert r['activation']==dict(mode='interval',start=6,end=14)
    assert p['activation']==f['activation']==dict(mode='endpoint',start=5,end=14)
    assert not chunk_candidate(p,6,32) and chunk_candidate(f,6,32)


def test_phase_publication_persists_fixed_reference_audit_and_recompiles_identically(tmp_path):
    path=phase_episode(tmp_path/'ep.hdf5');t=phase_template()
    with h5py.File(path) as h:cs,audit=instantiate(h,t)
    embed_review(path,dict(source_sha256=digest(path),constraints=cs),agent_run='phase-test',reference_template=t)
    with h5py.File(path) as h:
        assert read_embedded(h)['constraints']==cs
        coverage=json.loads(h['trajectory_constraints/coverage'][()])
        history=json.loads(h['trajectory_constraints/history'][()])[-1]
        assert coverage==json.loads(json.dumps(audit['coverage']))
        assert history['programmatic_audit']['reference_motion_phases']==audit['reference_motion_phases']
        again,again_audit=instantiate(h,t)
        assert again==cs and again_audit==audit
    before=digest(path)
    with pytest.raises(ValueError):embed_review(path,dict(source_sha256=before,constraints=cs),agent_run='no-template')
    assert digest(path)==before


def test_annotation_policy_fingerprint_binds_phase_helper_source_and_constants(tmp_path,monkeypatch):
    from services.robotwin import acceptance, reference_motion_phase
    root=Path(acceptance.__file__).parent
    # Actual isolated source bytes, not a fake source hash.
    for p in root.glob('*.py'):
        (tmp_path/p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(acceptance,'__file__',str(tmp_path/'acceptance.py'))
    before=acceptance.annotation_policy_fingerprint()
    helper=tmp_path/'reference_motion_phase.py'
    helper.write_bytes(helper.read_bytes()+b'\n# isolated source drift probe\n')
    assert acceptance.annotation_policy_fingerprint()!=before
    helper.write_bytes((root/'reference_motion_phase.py').read_bytes())
    assert acceptance.annotation_policy_fingerprint()==before
    monkeypatch.setitem(reference_motion_phase.POLICY,'stationary_step_m',2e-5)
    assert acceptance.annotation_policy_fingerprint()!=before
