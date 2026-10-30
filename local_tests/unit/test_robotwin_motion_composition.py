"""Ordinary-motion composition is stable under additional geometric sets."""
from copy import deepcopy
import json

import h5py
import numpy as np
import pytest

from services.robotwin.annotation import digest,embed_review
from services.robotwin.acceptance import trajectory_violations
from services.robotwin.constraint_schema import read_embedded
from services.robotwin.constraint_coverage import chunk_candidate,coverage_masks
from services.robotwin.full_trajectory import PREFIX,complete_constraints,motion_exemptions
from services.robotwin.programmatic_annotation import instantiate,legacy_motion_coverage,template_hash,validate_template
from services.robotwin.constraint_refinement import development_summary,propose_revision
from test_robotwin_programmatic_annotation import episode,template
from test_robotwin_constraint_refinement import row


def policy(t,exemptions=None,mode='independent'):
    return dict(version=1,task=t['task'],mode=mode,exemptions=exemptions or [],rationale='Explicit composition test')


def geometry(identifier='new_region',local_point=False):
    parameters=dict(axis=[1,0,0])
    if local_point:
        parameters['local_point']=dict(frame='subject_eef',offset_m=[.12,0,0],provenance=dict(source='manual',reference='synthetic fixed rigid point',sha256='f'*64,rationale='Geometry fixture, not native contact'))
    return dict(id=identifier,type='axis_position_interval',subjects='active_arms',
                selector=dict(kind='gripper_plateaus',state='closed',after='first_close'),
                origin='selection_start',orientation='world',parameters=parameters,
                allowed_margin=dict(lower=.02,upper=.02,unit='m'),rationale='Broad new region with free path inside.')


def guards(cs):return [c for c in cs if c['id'].startswith(PREFIX)]


@pytest.mark.parametrize('local_point',[False,True])
@pytest.mark.parametrize('migrate',[False,True])
def test_added_geometry_cannot_remove_or_change_existing_motion(tmp_path,local_point,migrate):
    t=template();t['motion_coverage']=legacy_motion_coverage(t) if migrate else policy(t)
    changed=deepcopy(t);changed['rules'].append(geometry(local_point=local_point))
    with h5py.File(episode(tmp_path/'ep.hdf5',seed=38),'r+') as h:
        h['endpose/left_gripper'][...] = np.r_[np.ones(2),np.zeros(44)]
        old,old_audit=instantiate(h,t);new,new_audit=instantiate(h,changed)
    assert guards(new)==guards(old)
    assert guards(new) and new_audit['coverage']['motion_composition']['complete']
    assert old_audit['coverage']['motion_composition']['replacement_reference_spans']==new_audit['coverage']['motion_composition']['replacement_reference_spans']
    # The old displacement is retained; its >32-step anchor conversion is unchanged.
    old_task={c['id']:{k:v for k,v in c.items() if k!='evidence'} for c in old if not c['id'].startswith(PREFIX)}
    assert all({k:v for k,v in c.items() if k!='evidence'}==old_task[c['id']] for c in new if c['id'] in old_task)
    assert next(c for c in new if c['id'].startswith('lift_'))['parameters']['anchor']=='reference_frame'
    for start in range(45):
        selected=[c for c in new if chunk_candidate(c,start,32)]
        coverage=coverage_masks(selected,46)
        assert all(all(coverage[a]['position_or_motion'][start+1:min(start+33,46)]) for a in ('left','right'))


def test_legacy_and_explicit_migration_preserve_exact_generated_constraints(tmp_path):
    t=template();explicit=deepcopy(t);explicit['motion_coverage']=policy(t,mode='legacy_position_override')
    migrated=deepcopy(t);migrated['motion_coverage']=legacy_motion_coverage(t)
    with h5py.File(episode(tmp_path/'ep.hdf5',seed=38),'r+') as h:
        h['endpose/left_gripper'][...] = np.r_[np.ones(2),np.zeros(44)]
        old,_=instantiate(h,t);legacy,legacy_audit=instantiate(h,explicit);new,new_audit=instantiate(h,migrated)
    assert guards(old)==guards(legacy)==guards(new)
    assert legacy_audit['coverage']['motion_composition']['policy']['mode']=='legacy_position_override'
    assert {e['rule_id'] for e in new_audit['coverage']['motion_composition']['resolved_exemptions']}=={'close','height','lift'}
    # Legacy still demonstrates the historical bug; compatibility is explicit.
    changed=deepcopy(t);changed['rules'].append(geometry())
    with h5py.File(tmp_path/'ep.hdf5') as h:replaced,_=instantiate(h,changed)
    assert guards(replaced)!=guards(old)


def test_allowed_set_intersection_keeps_a_real_direction_violation(tmp_path):
    t=template();legacy_add=deepcopy(t);legacy_add['rules'].append(geometry(local_point=True))
    migrated_add=deepcopy(legacy_add);migrated_add['motion_coverage']=legacy_motion_coverage(t)
    with h5py.File(episode(tmp_path/'ep.hdf5')) as h:
        old,_=instantiate(h,t);legacy,_=instantiate(h,legacy_add);new,_=instantiate(h,migrated_add)
        base=read_embedded(h)
        reference={a:dict(pose=h[f'endpose/{a}_endpose'][()].tolist(),gripper=h[f'endpose/{a}_gripper'][()].tolist()) for a in ('left','right')}
    candidate=deepcopy(reference);candidate['left']['pose'][3][2]=.5
    def accepted(cs):
        b=deepcopy(base);b['schema_version']='1.5';b['constraints']=cs
        b['provenance'].update(split='validation',variant='clean_heldout',source='acceptance_reference_not_for_training')
        return trajectory_violations(b,reference,candidate)['passed']
    assert not accepted(old)
    assert accepted(legacy)  # The original implicit replacement bug is observable.
    assert not accepted(new)
    assert guards(new)==guards(old)


@pytest.mark.parametrize('name',['a','a_b','a_left_12','a.b+[x]'])
def test_exemption_uses_exact_canonical_rule_identity_not_prefix(tmp_path,name):
    t=template();t['rules'][0]['id']=name
    t['motion_coverage']=legacy_motion_coverage(t)
    changed=deepcopy(t);changed['rules'].append(geometry(name+'_suffix'))
    with h5py.File(episode(tmp_path/'ep.hdf5')) as h:
        old,_=instantiate(h,t);new,audit=instantiate(h,changed)
    assert guards(new)==guards(old)
    exemption=next(e for e in audit['coverage']['motion_composition']['resolved_exemptions'] if e['rule_id']==name)
    assert all(c['id']==f'{name}_left_2_2' for c in exemption['compiled_constraints'])
    fake=deepcopy(next(c for c in new if c['id']==f'{name}_left_2_2'));fake['subject']='right_eef'
    with pytest.raises(ValueError,match='identity/subject'):
        motion_exemptions([fake],dict(exemptions=[t['motion_coverage']['exemptions'][0]]),8)


@pytest.mark.parametrize('change',['unknown','duplicate','cross_task','reserved','missing_provenance','fixed_frame','gripper','missing_primitive','changed_primitive'])
def test_invalid_exemption_declarations_fail_closed(change):
    t=template();t['motion_coverage']=legacy_motion_coverage(t);p=t['motion_coverage']
    if change=='unknown':p['exemptions'][0]['rule_id']='not_in_template'
    elif change=='duplicate':p['exemptions'].append(deepcopy(p['exemptions'][0]))
    elif change=='cross_task':p['task']='different_task'
    elif change=='reserved':t['rules'][0]['id']='__full_motion_v2__manual'
    elif change=='missing_provenance':del p['exemptions'][0]['provenance']
    elif change=='fixed_frame':p['exemptions'][0]['frames']=[1,2]
    elif change=='missing_primitive':del p['exemptions'][0]['primitive']
    elif change=='changed_primitive':p['exemptions'][0]['primitive']='axis_position_interval'
    else:p['exemptions'][0]['rule_id']='closed'
    with pytest.raises(ValueError):validate_template(t)


def test_direct_completion_needs_context_and_missing_reference_selection_fails(tmp_path):
    t=template();t['motion_coverage']=legacy_motion_coverage(t)
    with h5py.File(episode(tmp_path/'ep.hdf5')) as h:
        cs,_=instantiate(h,t)
        with pytest.raises(ValueError,match='task context'):
            complete_constraints(h,cs,'test',motion_coverage=t['motion_coverage'])
        with pytest.raises(ValueError,match='no compiled reference selection'):
            complete_constraints(h,[c for c in cs if not c['id'].startswith('close_')],'test',
                                 motion_coverage=t['motion_coverage'],task=t['task'])


def test_hdf5_recompile_is_idempotent_persists_audit_and_cannot_silently_downgrade(tmp_path):
    path=episode(tmp_path/'ep.hdf5');t=template();t['motion_coverage']=legacy_motion_coverage(t)
    t['coverage_policy']=dict(hold_margin_m=.04);t['rules'].append(geometry(local_point=True))
    with h5py.File(path) as h:cs,audit=instantiate(h,t)
    embed_review(path,dict(source_sha256=digest(path),constraints=cs),agent_run='test',reference_template=t)
    with h5py.File(path) as h:
        assert read_embedded(h)['constraints']==cs
        stored=json.loads(h['trajectory_constraints/coverage'][()])
        assert stored==json.loads(json.dumps(audit['coverage']))
        assert json.loads(h['trajectory_constraints/history'][()])[-1]['coverage']==stored
    before=digest(path)
    for legacy_template in (None,template()):
        with pytest.raises(ValueError,match='silent legacy downgrade'):
            embed_review(path,dict(source_sha256=before,constraints=cs),agent_run='no-downgrade',reference_template=legacy_template)
        assert digest(path)==before


def test_rsi_motion_migration_is_explicit_parent_bound_and_not_numeric_tuning():
    t=template();dev=development_summary([row()]);p=legacy_motion_coverage(t)
    patch=dict(version=1,base_template_sha256=template_hash(t),reason='Explicit migration only',
               evidence_ids=[dev['rows'][0]['trial_id']],operations=[dict(op='set_motion_coverage',policy=p)])
    revised=propose_revision(t,patch,dev)
    assert revised['template']['motion_coverage']==p and revised['template']['rules']==t['rules']
    for mutate in ['wrong_hash','missing_old_rule','numeric_op','duplicate_op']:
        bad=deepcopy(patch)
        if mutate=='wrong_hash':bad['operations'][0]['policy']['exemptions'][0]['provenance']['sha256']='0'*64
        elif mutate=='missing_old_rule':bad['operations'][0]['policy']['exemptions'].pop()
        elif mutate=='numeric_op':bad['operations'][0]['op']='set_coverage_policy'
        else:bad['operations']*=2
        with pytest.raises(ValueError):propose_revision(t,bad,dev)


def test_primitive_replacement_requires_an_explicit_reviewed_composition_change(tmp_path):
    t=template();t['motion_coverage']=legacy_motion_coverage(t)
    replacement=deepcopy(t['rules'][0]);replacement['type']='axis_position_interval'
    replacement['parameters']={'axis':[1,0,0]}
    dev=development_summary([row()])
    patch=dict(version=1,base_template_sha256=template_hash(t),reason='Test explicit primitive binding',
               evidence_ids=[dev['rows'][0]['trial_id']],operations=[dict(op='replace_rule',rule=replacement)])
    with pytest.raises(ValueError,match='identity and primitive'):
        propose_revision(t,patch,dev)
    changed=deepcopy(t['motion_coverage'])
    for e in changed['exemptions']:
        e['provenance']=dict(source='manual_review',reference='explicit alternative replacement semantics',sha256='e'*64)
        if e['rule_id']==replacement['id']:e['primitive']=replacement['type']
    patch['operations'].append(dict(op='set_motion_coverage',policy=changed))
    revised=propose_revision(t,patch,dev)['template']
    with h5py.File(episode(tmp_path/'ep.hdf5')) as h:
        cs,audit=instantiate(h,revised)
    assert audit['coverage']['motion_composition']['policy']==changed
    fake=deepcopy(next(c for c in cs if c['id'].startswith('close_')))
    fake['type']='point_distance_interval'
    with pytest.raises(ValueError,match='identity/subject'):
        motion_exemptions([fake],dict(exemptions=[changed['exemptions'][0]]),8)
