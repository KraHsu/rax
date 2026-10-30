"""New publication uses a versioned window objective; stored old labels stay valid."""
import copy
import json
import numpy as np
import pytest
import h5py

from services.robotwin.annotation import reference_values
from services.robotwin.constraint_schema import validate_constraint, validate_bundle, read_embedded
from services.robotwin.full_trajectory import complete_constraints
from services.dataset_viewer.store import feature_curve
from test_robotwin_programmatic_annotation import episode


def test_generated_windows_admit_reference_and_have_fixed_segment_fallback(tmp_path):
    path = episode(tmp_path/'episode.hdf5')
    with h5py.File(path) as h:
        cs, audit = complete_constraints(h, [], 'fixture')
        assert audit['complete'] and 'Temporal coverage only' in audit['meaning']
        windows = [c for c in cs if c['type'] == 'window_direction_cone']
        assert windows and audit['policy']['version'] == 3
        for c in windows:
            validate_constraint(c)
            assert c['parameters']['window_steps'] == 8
            assert c['parameters']['fallback_radius_m'] >= .02
            values = reference_values(h, c)
            assert np.all(values <= c['parameters']['bounds'][1]+1e-10)
            a = c['activation']; curve = feature_curve(h, c)
            np.testing.assert_allclose(curve[a['start']:a['end']], values)
            json.dumps(curve.tolist(), allow_nan=False)


def test_window_cannot_be_interpreted_as_an_old_bundle(tmp_path):
    path = episode(tmp_path/'episode.hdf5')
    with h5py.File(path) as h:
        bundle = read_embedded(h)
        cs, _ = complete_constraints(h, [], 'fixture')
    bundle.update(schema_version='1.3', constraints=cs); validate_bundle(bundle)
    old = copy.deepcopy(bundle); old['schema_version'] = '1.2'
    with pytest.raises(ValueError, match='schema 1.3'):
        validate_bundle(old)


def test_window_configuration_rejects_single_step_fallback_to_derivative(tmp_path):
    with h5py.File(episode(tmp_path/'episode.hdf5')) as h:
        cs, _ = complete_constraints(h, [], 'fixture')
    c = next(c for c in cs if c['type'] == 'window_direction_cone')
    c['parameters']['minimum_window_steps'] = 1
    with pytest.raises(ValueError, match='minimum_window_steps'):
        validate_constraint(c)
