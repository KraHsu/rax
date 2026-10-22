"""One native episode, command budget, measured observations and operator record."""
from pathlib import Path
import json
import uuid

import numpy as np

from services.robotwin.tools import CAMERAS, validate_request
from services.simulator.io import action_execution_scope, reject_unless, write_json


class RoboTwinSession:
    def __init__(self, factory, *, task, instruction, output, native_step_limit, provenance, recording_path=None):
        self.factory, self.task, self.instruction = factory, task, instruction
        self.output, self.limit, self.provenance = Path(output), native_step_limit, provenance
        self.native = None
        self.episode_id, self.step_id = None, 0
        self.terminated = self.truncated = False
        self.invalid_reason = self.finish_reason = None
        self.success = False
        self.recording_path, self.recorder = recording_path, None
        self.metadata = {'task': task, 'benchmark': 'robotwin', 'native_step_limit': self.limit,
                         'eef_frame': 'world', 'eef_quaternion': 'wxyz', 'action_mode': 'absolute_targets',
                         'command_duration': 'native_planner_variable'}

    def _observation(self):
        raw = self.native.get_obs()
        # Explicit whitelist: no actor poses, segmentation, seeds, success flags
        # or camera calibration can enter the agent's RGB-only observations.
        observation = {'episode_id': self.episode_id, 'step_id': self.step_id, 'instruction': self.instruction,
                       'endpose': {key: np.asarray(raw['endpose'][key]).tolist() for key in
                                   ('left_endpose', 'left_gripper', 'right_endpose', 'right_gripper')},
                       'joint_action': {'vector': np.asarray(raw['joint_action']['vector']).tolist()},
                       'images': {}, 'remaining_commands': max(0, self.limit-self.step_id),
                       'recorded_frame': self.step_id}
        for camera in CAMERAS:
            pixels = raw.get('observation', {}).get(camera, {}).get('rgb')
            if pixels is not None:
                observation['images'][camera] = np.asarray(pixels, dtype=np.uint8)
        return observation

    def _write_summary(self, reason=None):
        write_json(self.output / 'operator_state.json', {
            'episode_id': self.episode_id, 'step_id': self.step_id, 'task': self.task,
            'terminated': self.terminated, 'truncated': self.truncated,
            'invalid_reason': self.invalid_reason, 'reason': reason,
            'reward': {'native_success': bool(self.success and self.invalid_reason is None)},
            'provenance': self.provenance})

    def dispatch(self, op, args):
        if op == 'metadata':
            return self.metadata
        if op == 'reset':
            reject_unless(self.native is None, 'One episode per process; cannot reset an existing scene')
            self.native = self.factory()
            self.episode_id = uuid.uuid4().hex
            if self.recording_path is not None:
                from services.robotwin.recording import EpisodeRecorder
                self.recorder = EpisodeRecorder(self.recording_path, recording_id=self.episode_id, instruction=self.instruction)
                self.recorder.append(self._observation())
            self._write_summary()
            return dict(episode_id=self.episode_id, step_id=0, native_step_limit=self.limit)
        reject_unless(self.native is not None and args.get('episode_id') == self.episode_id
                      and type(args.get('step_id')) is int and args['step_id'] == self.step_id,
                      'Missing or stale episode/step identity')
        if op == 'observe':
            return self._observation()
        if op == 'operator_state':
            # Trusted host RPC only; never listed in the agent tool contract.
            return json.loads((self.output / 'operator_state.json').read_text())
        if op == 'constraints_list':
            result = self.recorder.annotations() if self.recorder else {'revision': None, 'constraints': [], 'recording': False}
            return dict(result, episode_id=self.episode_id, step_id=self.step_id)
        if op == 'constraint_upsert':
            reject_unless(self.recorder is not None and self.provenance.get('collection') == 0,
                          'Constraint updates require a clean training recording')
            result = self.recorder.upsert(args.get('constraint'), args.get('expected_revision'))
            return dict(result, episode_id=self.episode_id, step_id=self.step_id)
        if op == 'finish_pilot':
            self.finish_reason = str(args.get('reason', 'finish'))
            self.truncated = not self.terminated
            self._write_summary(self.finish_reason)
            if self.recorder is not None:
                self.recorder.finish(success=self.success and self.invalid_reason is None)
            return {'episode_id': self.episode_id, 'step_id': self.step_id, 'finished': True}
        reject_unless(op == 'chunk_step', 'Unknown native operation')
        reject_unless(not (self.terminated or self.truncated), 'Episode ended')
        mode = args.get('action_type')
        reject_unless(mode in ('qpos', 'ee'), 'action_type must be qpos or ee')
        rows = args.get('actions')
        validate_request('robotwin_step_eef' if mode == 'ee' else 'robotwin_step', {'actions': rows})
        before = self._observation()
        if mode == 'qpos':
            reject_unless(len(rows[0]) == len(before['joint_action']['vector']), 'Joint width does not match this robot')
        executed = 0
        with action_execution_scope(self, 'chunk_step'):
            for row in rows:
                if self.terminated or self.truncated:
                    break
                previous = self.step_id
                try:
                    self.native.take_action(np.asarray(row, dtype=np.float64), action_type=mode)
                    self.step_id = int(self.native.take_action_cnt)
                    if self.step_id != previous+1:
                        raise RuntimeError('Native command count did not advance exactly once')
                    executed += 1
                    self.success = bool(self.native.eval_success)
                    self.terminated = self.success
                    self.truncated = self.step_id >= self.limit and not self.success
                    after = self._observation()
                    if self.recorder is not None:
                        self.recorder.append(after)
                    write_json(self.output / 'transitions' / f'{self.step_id:06d}.json', {
                        'episode_id': self.episode_id, 'start_step': previous, 'end_step': self.step_id,
                        'action_type': mode, 'action': row,
                        'before': {k: v for k, v in before.items() if k != 'images'},
                        'after': {k: v for k, v in after.items() if k != 'images'}})
                    before = after
                except Exception:
                    self.step_id = int(self.native.take_action_cnt)
                    self.invalid_reason, self.truncated = 'native_execution_error', True
                    raise
                finally:
                    self._write_summary(self.invalid_reason)
        return {'observation': before, 'executed_rows': executed, 'episode_id': self.episode_id,
                'step_id': self.step_id, 'stopped': self.terminated or self.truncated}

    def close(self):
        try:
            if self.recorder is not None and self.recorder.handle is not None:
                self.recorder.finish(success=False)
        finally:
            if self.native is not None:
                self.native.close_env()
