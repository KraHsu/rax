"""Read existing predicate locals; never invoke or replace the native predicate."""
from __future__ import annotations

import hashlib
import sys
import threading
from pathlib import Path

import numpy as np

TASK_SOURCE_SHA256 = '53028cfe0406971327ef6ad294191f3d32340952246cdf0085598f35b910d94a'
MAX_PREDICATE_ROWS = 100000
MAX_ROOT_POSE_ROWS = 5000
MAX_ERRORS = 128


def pack(value):
    """Preserve numpy scalar/container types for later predicate recomputation."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'fiu' or not np.isfinite(value).all():
            raise ValueError('Invalid numeric diagnostic array')
        return dict(kind='ndarray', dtype=str(value.dtype), shape=list(value.shape), value=value.tolist())
    if isinstance(value, np.generic):
        if value.dtype.kind not in 'fbiu' or not np.isfinite(value):
            raise ValueError('Invalid numeric diagnostic scalar')
        return dict(kind='numpy_scalar', dtype=str(value.dtype), value=value.item())
    if type(value) in (list, tuple):
        return dict(kind=type(value).__name__, value=[pack(x) for x in value])
    if type(value) in (float, int, bool):
        if not np.isfinite(value):
            raise ValueError('Nonfinite diagnostic value')
        return dict(kind=type(value).__name__, value=value)
    if value is None:
        return dict(kind='none')
    raise ValueError('Unsupported diagnostic type: ' + type(value).__name__)


def unpack(item):
    kind = item['kind']
    if kind == 'ndarray':
        value = np.asarray(item['value'], dtype=np.dtype(item['dtype']))
        if list(value.shape) != item['shape'] or not np.isfinite(value).all():
            raise ValueError('Diagnostic array changed')
        return value
    if kind == 'numpy_scalar':
        return np.dtype(item['dtype']).type(item['value'])
    if kind in ('list', 'tuple'):
        value = [unpack(x) for x in item['value']]
        return value if kind == 'list' else tuple(value)
    if kind in ('int', 'float', 'bool'):
        return {'int': int, 'float': float, 'bool': bool}[kind](item['value'])
    if kind == 'none':
        return None
    raise ValueError('Unknown diagnostic kind')


def components(row):
    """Match source expression order and strict boundaries; diagnostic only."""
    a, b = unpack(row['bottle1_pose']), unpack(row['bottle2_pose'])
    ta, tb = unpack(row['bottle1_target']), unpack(row['bottle2_target'])
    checks = [abs(a[0]-ta[0]) < .1, abs(a[1]-ta[1]) < .1, a[2] > .89,
              abs(b[0]-tb[0]) < .1, abs(b[1]-tb[1]) < .1, b[2] > .89]
    return dict(per_component=[bool(x) for x in checks], success=bool(all(checks)),
                actor_pose_forced_false=bool(a[2] < .78 or b[2] < .78))


class PredicateObserver:
    def __init__(self, native, source_path):
        path = Path(source_path).resolve()
        method = native.check_success
        code = method.__func__.__code__
        if (hashlib.sha256(path.read_bytes()).hexdigest() != TASK_SOURCE_SHA256
                or Path(code.co_filename).resolve() != path or code.co_name != 'check_success'):
            raise ValueError('Predicate observer requires exact original task source/method')
        self.native, self.code = native, code
        self.rows, self.errors, self.poses = [], [], []
        self.calls = 0
        self.command_frame = None
        self.phase = 'setup_complete'
        self.pending = {}
        self.owner_thread = threading.get_ident()
        self.installed_callback = None
        self.errors_total = 0
        self.omitted_predicate_returns = 0
        self.omitted_root_poses = 0
        self.last_return = None
        self.original_profile = None
        self.installed = False

    def _error(self, where, exc):
        self.errors_total += 1
        if len(self.errors) < MAX_ERRORS:
            self.errors.append(dict(where=where, call=self.calls, command_frame=self.command_frame,
                                    error=type(exc).__name__ + ': ' + str(exc)))

    def _profile(self, frame, event, arg):
        if frame.f_code is not self.code:
            return
        # Never call getters, the predicate, physics or CUDA from this callback.
        try:
            if frame.f_locals.get('self') is not self.native:
                return
            if event == 'call':
                self.pending[id(frame)] = pack(vars(self.native).get('actor_pose'))
            elif event == 'return':
                self.calls += 1
                before = self.pending.pop(id(frame))
                data = dict(phase=self.phase, command_frame=self.command_frame,
                            caller_function=frame.f_back.f_code.co_name if frame.f_back else None,
                            caller_file=frame.f_back.f_code.co_filename if frame.f_back else None,
                            native_take_action_cnt=vars(self.native).get('take_action_cnt'),
                            actor_pose_before=before, actor_pose_after=pack(vars(self.native).get('actor_pose')),
                            **{key: pack(frame.f_locals[key]) for key in
                               ('bottle1_pose', 'bottle2_pose', 'bottle1_target', 'bottle2_target')})
                if type(arg) not in (bool, np.bool_):
                    raise ValueError('Native predicate did not return a boolean')
                data['native_result'] = bool(arg)
                if components(data)['success'] is not data['native_result']:
                    raise ValueError('Predicate components disagree with original return')
                self.last_return = dict(call=self.calls, data=data)
                if self.rows and self.rows[-1]['last_call'] == self.calls-1 and self.rows[-1]['data'] == data:
                    self.rows[-1]['last_call'] = self.calls
                elif len(self.rows) < MAX_PREDICATE_ROWS:
                    self.rows.append(dict(first_call=self.calls, last_call=self.calls, data=data))
                else:
                    self.omitted_predicate_returns += 1
                    if self.omitted_predicate_returns == 1:
                        self._error('predicate_capacity', ValueError('Fixed diagnostic row capacity exceeded'))
        except Exception as exc:
            self._error('predicate_profile', exc)

    def install(self):
        if self.installed or sys.getprofile() is not None:
            raise ValueError('Cannot replace another thread profiler')
        self.original_profile = sys.getprofile()
        self.installed_callback = self._profile
        sys.setprofile(self.installed_callback)
        self.installed = True

    def close(self):
        if self.installed:
            if threading.get_ident() != self.owner_thread:
                self._error('owner_thread_changed', RuntimeError('Close called outside profile owner thread'))
                raise RuntimeError('Cannot restore another thread profiler')
            if sys.getprofile() is not self.installed_callback:
                self._error('profile_lost', RuntimeError('Installed profile callback was replaced'))
                # Do not clobber the replacement profile; evidence is incomplete.
            else:
                sys.setprofile(self.original_profile)
            self.installed = False
            if self.phase == 'replay_existing_predicate_cadence':
                last = self.last_return
                if (last is None or last['data']['caller_function'] != 'replay'
                        or not last['data']['caller_file'].endswith('/services/robotwin/acceptance_replay.py')):
                    self._error('missing_original_replay_final', ValueError('No original replay final predicate return observed'))

    def observe_pose(self, label, frame=None):
        """Only read root poses at explicit capture samples/command boundaries."""
        try:
            if len(self.poses) >= MAX_ROOT_POSE_ROWS:
                self.omitted_root_poses += 1
                if self.omitted_root_poses == 1:
                    self._error('pose_capacity', ValueError('Fixed diagnostic pose capacity exceeded'))
                return
            row = dict(label=label, frame=frame, phase=self.phase, predicate_calls=self.calls, actors={})
            for name in ('bottle1', 'bottle2'):
                pose = getattr(self.native, name).get_pose()
                row['actors'][name] = dict(p=pack(pose.p), q_wxyz=pack(pose.q))
            self.poses.append(row)
        except Exception as exc:
            self._error('root_pose', exc)

    def result(self):
        return dict(version=1, purpose='diagnostic_only_not_constraint_or_training_input',
                    numeric_context=dict(numpy_version=np.__version__, comparison='original_source_scalar_promotion'),
                    source_sha256=TASK_SOURCE_SHA256, predicate_calls=self.calls,
                    predicate_call_count_scope='observed returns on owner thread; incomplete monitoring is not a total native call count',
                    predicate_rle=self.rows, root_pose_samples=self.poses, errors=self.errors,
                    owner_thread=self.owner_thread, thread_scope='current_owner_thread_only',
                    errors_total=self.errors_total, last_return=self.last_return,
                    omitted_predicate_returns=self.omitted_predicate_returns,
                    omitted_root_poses=self.omitted_root_poses,
                    capacity=dict(predicate_rows=MAX_PREDICATE_ROWS, root_pose_rows=MAX_ROOT_POSE_ROWS, errors=MAX_ERRORS),
                    observation_complete=self.errors_total == 0 and not self.pending,
                    extra_predicate_calls=0, control_or_predicate_replaced=False,
                    record_rule='existing original predicate calls only; contiguous RLE retains every return and phase/frame; root getters only at command/capture-sample boundaries')


def verify_observations(value):
    if value['numeric_context'] != dict(numpy_version=np.__version__, comparison='original_source_scalar_promotion'):
        raise ValueError('Predicate recomputation requires the original NumPy numeric context')
    if (value['source_sha256'] != TASK_SOURCE_SHA256 or value['extra_predicate_calls'] != 0
            or value['control_or_predicate_replaced'] is not False):
        raise ValueError('Observer source/cadence changed')
    if (value['thread_scope'] != 'current_owner_thread_only'
            or value['errors_total'] < len(value['errors']) or len(value['errors']) > MAX_ERRORS
            or len(value['predicate_rle']) > MAX_PREDICATE_ROWS or len(value['root_pose_samples']) > MAX_ROOT_POSE_ROWS):
        raise ValueError('Diagnostic scope/capacity changed')
    cursor, successes, recorded = 1, 0, 0
    for row in value['predicate_rle']:
        if (row['first_call'] < cursor or type(row['last_call']) is not int or row['last_call'] < row['first_call']
                or (value['observation_complete'] and row['first_call'] != cursor)):
            raise ValueError('Predicate timeline gap/overlap')
        check = components(row['data'])
        if check['success'] is not row['data']['native_result']:
            raise ValueError('Predicate result changed')
        if check['actor_pose_forced_false'] and unpack(row['data']['actor_pose_after']) is not False:
            raise ValueError('Original actor_pose side effect missing')
        n = row['last_call']-row['first_call']+1
        successes += n * int(check['success'])
        recorded += n
        cursor = row['last_call']+1
    if value['observation_complete'] and (value['errors_total'] or cursor != value['predicate_calls']+1
            or value['omitted_predicate_returns'] or value['omitted_root_poses']):
        raise ValueError('Complete predicate observations have a gap')
    if cursor-1 > value['predicate_calls']:
        raise ValueError('Predicate observations exceed original call count')
    last = value['last_return']
    if last is not None and (last['call'] > value['predicate_calls'] or components(last['data'])['success'] is not last['data']['native_result']):
        raise ValueError('Final recorded return is invalid')
    if value['observation_complete'] and value['predicate_calls']:
        if (last is None or last['call'] != value['predicate_calls']
                or not value['predicate_rle'] or last['data'] != value['predicate_rle'][-1]['data']):
            raise ValueError('Complete final return differs from full predicate timeline')
        if last['data']['phase'] == 'replay_existing_predicate_cadence' and (
                last['data']['caller_function'] != 'replay'
                or not last['data']['caller_file'].endswith('/services/robotwin/acceptance_replay.py')):
            raise ValueError('Complete replay lacks its original final predicate call')
    return dict(recorded_calls=recorded, successful_calls=successes,
                any_recorded_success=successes > 0, observation_complete=value['observation_complete'])
