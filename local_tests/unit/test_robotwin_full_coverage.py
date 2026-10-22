import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from services.robotwin.annotation import digest, reference_values, validate_reference_bounds
from services.robotwin.constraint_schema import read_embedded, validate_bundle
from services.robotwin.constraint_coverage import coverage_report, coverage_masks, chunk_candidate
from services.robotwin.full_trajectory import complete_constraints, PREFIX
from services.dataset_viewer.store import feature_curve
from scripts.backfill_robotwin_motion_coverage import upgrade
from test_robotwin_programmatic_annotation import episode, instantiate, template


def generated(path):
    with h5py.File(path, 'r') as h:
        b = read_embedded(h)
        cs, report = complete_constraints(h, b['constraints'], b['provenance']['recording_id'])
        b.update(constraints=cs, schema_version='1.1')
        validate_bundle(b, len(h['endpose/left_endpose'])); validate_reference_bounds(b, h)
        return cs, report


def test_full_coverage_splits_turns_and_stops_without_absolute_moving_targets(tmp_path):
    path = episode(tmp_path/'ep.hdf5')
    positions = np.array([[0,0,0],[.01,0,0],[.02,.004,0],[.03,.01,0],
                          [.03,.01,0],[.03,.01,0],[.03,.02,0],[.03,.03,0]])
    with h5py.File(path, 'r+') as h:
        poses = h['endpose/left_endpose'][:]; poses[:,:3] = positions; h['endpose/left_endpose'][:] = poses
    cs, report = generated(path)
    assert report['complete'] and report['covered'] == 7
    assert report['per_arm']['left']['orientation']['covered'] == 0  # Free rotation is not silently constrained.
    left = [c for c in cs if c['subject'] == 'left_eef']
    assert [c['type'] for c in left] == ['motion_direction_cone', 'point_distance_interval', 'motion_direction_cone']
    assert [c['activation'] for c in left] == [dict(mode='interval', start=a, end=b) for a,b in [(1,4),(4,6),(6,8)]]
    assert left[0]['parameters']['axis'] == [1.,0.,0.] and left[-1]['parameters']['axis'] == [0.,1.,0.]
    # Every possible short chunk, including ones wholly inside a long interval, has geometry for both arms.
    for start in range(7):
        for horizon in (1,2,4):
            candidates = [c for c in cs if chunk_candidate(c, start, min(horizon, 7-start))]
            masks = coverage_masks(candidates, 8)
            for arm in ('left','right'): assert all(masks[arm]['position_or_motion'][start+1:min(8,start+horizon+1)])


def test_reference_specific_features_take_precedence_and_regeneration_is_idempotent(tmp_path):
    path = episode(tmp_path/'ep.hdf5')
    with h5py.File(path, 'r') as h:
        cs, audit = instantiate(h, template())
        assert audit['coverage']['complete']
        generated_constraints = [c for c in cs if c['id'].startswith(PREFIX)]
        assert not any(c['subject']=='left_eef' and c['activation']['start'] <= 2 < c['activation']['end'] for c in generated_constraints)
        again, _ = complete_constraints(h, cs, 'ref0')
        assert again == cs
        for c in generated_constraints:
            active = range(c['activation']['start'],c['activation']['end'])
            np.testing.assert_allclose(feature_curve(h,c)[list(active)], reference_values(h,c))


def test_gripper_only_or_old_displacement_does_not_fake_geometry_coverage(tmp_path):
    path = episode(tmp_path/'ep.hdf5')
    with h5py.File(path, 'r') as h: cs, _ = instantiate(h, template())
    sparse = [c for c in cs if c['type'] in ('gripper_interval','axis_displacement_interval')]
    report = coverage_report(sparse,8)
    assert not report['complete'] and report['covered'] == 0


def test_backfill_preserves_actions_and_is_repeatable(tmp_path):
    path = episode(tmp_path/'ep.hdf5')
    with h5py.File(path) as h: before = {k:h['endpose/'+k][:] for k in h['endpose']}
    result = upgrade(path); assert result['coverage']['complete']
    with h5py.File(path) as h:
        assert read_embedded(h)['schema_version']=='1.1'
        for k,v in before.items(): np.testing.assert_array_equal(h['endpose/'+k][:],v)
        assert json.loads(h['trajectory_constraints/coverage'][()])['complete']
    checksum = digest(path)
    assert upgrade(path) is None and digest(path)==checksum


def test_new_primitive_is_versioned_and_shared_with_training():
    root = Path(__file__).resolve().parents[2]
    other = root.parent/'OpenWAM/openwam/train'
    if other.exists():
        for name in ('constraint_schema.py','constraint_coverage.py'):
            assert (root/'services/robotwin'/name).read_bytes()==(other/name).read_bytes()
