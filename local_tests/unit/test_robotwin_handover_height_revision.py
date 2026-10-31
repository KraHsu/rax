"""The one named migration must reject lookalikes and preserve compiled masks."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from services.robotwin.constraint_schema import HDF5_KEY
from services.robotwin.handover_mic_height_revision import (
    HeightRevisionError, PARENT_SHA256, audit_reference, candidate_template, verify_compiled_pair,
)
from services.robotwin.programmatic_annotation import instantiate, template_hash


def parent():
    return json.loads((Path(__file__).parents[1]/'fixtures/handover_mic_height_revision/parent.json').read_text())


class Handle(dict):
    pass


def reference(*, release=190, close=40, length=220, donor='left'):
    h = Handle()
    h.attrs = {'robotwin_expert_provenance': json.dumps({'task': 'handover_mic'})}
    h[HDF5_KEY] = np.asarray(json.dumps(dict(schema_version='1.5', constraints=[],
        action_contract=dict(position='absolute', rotation='absolute', quaternion_order='wxyz',
                             gripper='absolute_open_fraction', frame='world'),
        provenance=dict(recording_id='synthetic-duplicate-height', source='synthetic', split='train',
                        variant='clean_50', native_success=True))))
    for side in ['left', 'right']:
        pose = np.tile([0., 0., .8, 1., 0., 0., 0.], (length, 1))
        pose[:, 0] = np.linspace(-.1, .1, length)
        pose[:, 2] += np.linspace(0, .2, length)
        grip = np.ones(length)
        grip[close if side == donor else close+10:] = 0
        if side == donor:
            grip[release:] = 1
        h[f'endpose/{side}_endpose'] = pose
        h[f'endpose/{side}_gripper'] = grip
    return h


def pair(h):
    compiled, _ = instantiate(h, parent())
    return [next(c for c in compiled if c['id'].startswith(prefix))
            for prefix in ['release_vertical_rise_', 'release_height_']]


@pytest.mark.parametrize('donor', ['left', 'right'])
def test_named_migration_compiles_all_retained_physics_and_every_chunk_identically(donor):
    p = parent(); original = copy.deepcopy(p); h = reference(donor=donor)
    revised, report = audit_reference(h, p, reference_id='synthetic')
    assert template_hash(p) == PARENT_SHA256 and p == original
    assert [r['id'] for r in revised['rules']] == [r['id'] for r in p['rules'] if r['id'] != 'release_vertical_rise']
    assert report['removed_to_retained']['subject'] == donor+'_eef'
    assert report['removed_to_retained']['eligible_chunk_starts'] == list(range(158, 190))
    assert report['automatic_motion_guards_identical'] and report['coverage_identical']
    assert report['candidate_only'] and not report['activated']


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(task='other'),
    lambda p: p.update(rationale='different parent'),
    lambda p: p['rules'][-1]['parameters'].update(axis=[1, 0, 0]),
])
def test_exact_parent_identity_required(mutation):
    p = parent(); mutation(p)
    with pytest.raises(HeightRevisionError):
        candidate_template(p)


def test_short_span_predicted_origin_is_not_duplicated_fixed_height():
    h = reference(close=150, release=170)
    rise, height = pair(h)
    assert rise['parameters'].get('anchor', 'predicted_start') == 'predicted_start'
    with pytest.raises(HeightRevisionError, match='fixed-reference'):
        audit_reference(h, parent(), reference_id='short-span')


@pytest.mark.parametrize('mutation', [
    lambda a, b: a['parameters'].update(anchor='predicted_start'),
    lambda a, b: b['parameters']['bounds'].__setitem__(1, b['parameters']['bounds'][1]+1e-12),
    lambda a, b: b['penalty'].update(scale=.01),
    lambda a, b: b['penalty'].update(weight=2),
    lambda a, b: b['activation']['event'].update(frame=191),
    lambda a, b: b.update(subject='right_eef'),
    lambda a, b: a['reference_frame'].update(quaternion_wxyz=[0., 1., 0., 0.]),
    lambda a, b: a['parameters'].update(local_point={'anything': 'unsupported'}),
])
def test_near_equal_or_different_semantics_are_never_implicitly_deduplicated(mutation):
    a, b = pair(reference()); mutation(a, b)
    with pytest.raises(HeightRevisionError):
        verify_compiled_pair(a, b, 220)


def test_existing_embedded_revision_preserves_historical_guards_and_extra_labels_verbatim():
    from services.robotwin.handover_mic_height_revision import candidate_embedded_bundle
    h=reference(); cs,_=instantiate(h,parent())
    b=json.loads(h[HDF5_KEY].item());b['constraints']=cs
    # Old generated datasets can have a different guard generation and extra
    # labels. A current compiler reannotation must not silently replace those.
    for c in b['constraints']:
        c['id']=c['id'].replace('__full_motion_v2__','__full_motion_v1__')
    extra=copy.deepcopy(cs[0]);extra['id']='historical_extra';b['constraints'].append(extra)
    original=copy.deepcopy(b)
    revised,report=candidate_embedded_bundle(b,220,parent=parent())
    assert b==original and report['legacy_motion_guards_preserved']
    assert revised['constraints']==[c for c in b['constraints'] if not c['id'].startswith('release_vertical_rise_')]
    assert {k:v for k,v in revised.items() if k!='constraints'}=={k:v for k,v in b.items() if k!='constraints'}
    revised['constraints'][0]['penalty']['weight']=2
    assert b==original  # No aliased original structures.


def test_existing_bundle_rejects_missing_or_extra_named_pairs():
    from services.robotwin.handover_mic_height_revision import candidate_embedded_bundle
    h=reference(); cs,_=instantiate(h,parent());b=json.loads(h[HDF5_KEY].item());b['constraints']=cs
    b['constraints']=[c for c in cs if not c['id'].startswith('release_height_')]
    with pytest.raises(HeightRevisionError,match='exactly'):
        candidate_embedded_bundle(b,220,parent=parent())
