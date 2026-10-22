"""Record native observations and base constraints into the same HDF5 episode.

Successful recordings are published under data/. Failed/interrupted episodes
remain under rejected/ and cannot silently become successful demonstrations.
"""
import copy
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from services.robotwin.constraint_schema import HDF5_KEY, SCHEMA_VERSION, canonical_json, validate_constraint, validate_bundle, validate_reference_events


class EpisodeRecorder:
    def __init__(self, path, *, recording_id, instruction):
        import h5py
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError('Recording already exists')
        self.temporary = self.path.with_name('.'+self.path.name+'.partial')
        self.handle = h5py.File(self.temporary, 'x')
        self.handle.attrs['instruction'] = instruction
        self.length, self.revision, self.history = 0, 0, []
        self.cameras = None
        self.bundle = {'schema_version': SCHEMA_VERSION, 'action_contract': {
            'position': 'absolute', 'rotation': 'absolute', 'quaternion_order': 'wxyz',
            'gripper': 'absolute_open_fraction', 'frame': 'world'},
            'provenance': {'split': 'train', 'variant': 'clean_50', 'recording_id': recording_id,
                           'source': 'rax_robotwin', 'sampling': 'native_command_boundaries_variable_duration'},
            'constraints': []}
        self._persist_annotations()

    def _persist_annotations(self):
        import h5py
        # Live future intervals may extend beyond the recorded prefix. Final
        # publication validates every interval against the completed episode.
        payload = canonical_json(self.bundle)
        for key, value in ((HDF5_KEY, payload), ('trajectory_constraints/history', json.dumps(self.history))):
            if key in self.handle:
                self.handle[key][()] = value
            else:
                self.handle.create_dataset(key, data=value, dtype=h5py.string_dtype('utf-8'))
        self.handle['trajectory_constraints'].attrs['revision'] = self.revision
        self.handle.flush()

    def annotations(self):
        return {'revision': self.revision, 'recorded_frames': self.length, 'constraints': copy.deepcopy(self.bundle['constraints'])}

    def upsert(self, constraint, expected_revision):
        if type(expected_revision) is not int or expected_revision != self.revision:
            raise ValueError(f'Stale annotation revision; current={self.revision}')
        validate_constraint(constraint)
        record = copy.deepcopy(constraint)
        constraints = [c for c in self.bundle['constraints'] if c['id'] != record['id']]
        constraints.append(record)
        self.bundle['constraints'] = constraints
        self.revision += 1
        self.history.append({'revision': self.revision, 'recorded_frames': self.length, 'constraint': record})
        self._persist_annotations()
        return self.annotations()

    def _append(self, key, value, *, jpeg=False):
        import h5py
        value = np.asarray(value)
        if key not in self.handle:
            if self.length:
                raise ValueError('Observation fields changed within one recording')
            if jpeg:
                self.handle.create_dataset(key, shape=(0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
            else:
                self.handle.create_dataset(key, shape=(0,)+value.shape, maxshape=(None,)+value.shape, dtype=value.dtype)
        dataset = self.handle[key]
        dataset.resize(self.length+1, axis=0)
        dataset[self.length] = value

    def append(self, observation):
        cameras = set(observation['images'])
        if not cameras or (self.cameras is not None and self.cameras != cameras):
            raise ValueError('Camera set must be nonempty and stable within an episode')
        self.cameras = cameras
        for arm in ('left', 'right'):
            pose = np.asarray(observation['endpose'][arm+'_endpose'], dtype=float)
            grip = float(observation['endpose'][arm+'_gripper'])
            if (pose.shape != (7,) or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-4)
                    or not -1e-5 <= grip <= 1+1e-5):
                raise ValueError('Expected finite measured XYZ + unit WXYZ and normalized opening')
        if not np.isfinite(observation['joint_action']['vector']).all():
            raise ValueError('Non-finite measured joints')
        for key in ('left_endpose', 'right_endpose', 'left_gripper', 'right_gripper'):
            self._append('endpose/'+key, np.asarray(observation['endpose'][key], dtype=np.float64))
        self._append('joint_action/vector', np.asarray(observation['joint_action']['vector'], dtype=np.float64))
        for camera, pixels in observation['images'].items():
            stream = io.BytesIO()
            # Match native RoboTwin's cv2.imencode(RGB) channel convention.
            Image.fromarray(np.asarray(pixels, dtype=np.uint8)[..., ::-1]).save(stream, format='JPEG', quality=95)
            self._append(f'observation/{camera}/rgb', np.frombuffer(stream.getvalue(), dtype=np.uint8), jpeg=True)
        self.length += 1
        self.handle.flush()

    def finish(self, *, success):
        if self.handle is None:
            return
        try:
            validate_bundle(self.bundle, self.length)
            validate_reference_events(self.bundle, self.handle)
            self.bundle['provenance']['native_success'] = bool(success)
            if success:
                import h5py
                from services.robotwin.full_trajectory import complete_constraints
                from services.robotwin.annotation import validate_reference_bounds
                self.bundle['constraints'], coverage = complete_constraints(
                    self.handle, self.bundle['constraints'], self.bundle['provenance']['recording_id'])
                validate_bundle(self.bundle, self.length)
                validate_reference_bounds(self.bundle, self.handle)
                self.revision += 1
                self.history.append({'revision': self.revision, 'agent_run': 'full-motion-coverage-v1',
                                     'coverage': coverage, 'constraints': copy.deepcopy(self.bundle['constraints'])})
                self.handle.create_dataset('trajectory_constraints/coverage', data=json.dumps(coverage), dtype=h5py.string_dtype())
            self._persist_annotations()
        finally:
            self.handle.close()
            self.handle = None
        target = self.path if success else self.path.parent.parent / 'rejected' / self.path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.rename(target)
        return str(target)
