"""Passive control-time diagnostics, with an explicit audited Python-call scope.

No simulation, tensor transfer or extra state getter is performed by a profile
callback. SAPIEN's System.step does not emit c_profile events in the supported
runtime, so successful calls are counted by a transparent declaring-class shim.
Cached original bound methods are outside that shim: source audit must exclude
such aliases, and the receipt never claims arbitrary-code coverage.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import tokenize

import numpy as np

VERSION = 'audited_backend_step_observer_v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def cpu_value(value):
    """Never call repr, conversion hooks, getters, tensor.cpu or synchronize."""
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value, True
    if kind is float:
        return (value, True) if np.isfinite(value) else ({'unobserved': 'nonfinite_float'}, False)
    if kind is np.ndarray:
        if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
            return {'unobserved': 'unsupported_numpy_dtype_or_nonfinite'}, False
        copy = value.copy(order='C')
        return dict(dtype=copy.dtype.str, shape=list(copy.shape),
                    bytes_sha256=hashlib.sha256(copy.tobytes()).hexdigest(), values=copy.tolist()), True
    if isinstance(value, np.generic) and value.dtype.kind in 'biuf':
        if not np.isfinite(value):
            return {'unobserved': 'nonfinite_numpy_scalar'}, False
        return dict(dtype=value.dtype.str, bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                    value=value.item()), True
    if kind in (list, tuple):
        items = [cpu_value(v) for v in value]
        return {'container': kind.__name__, 'items': [v for v, _ in items]}, all(ok for _, ok in items)
    if kind is dict and all(type(k) is str for k in value):
        items = {k: cpu_value(v) for k, v in sorted(value.items())}
        return {k: v for k, (v, _) in items.items()}, all(ok for _, ok in items.values())
    return {'unobserved_type': f'{kind.__module__}.{kind.__qualname__}'}, False


def source_scope(paths, *, observed_timestep_setter=None, observed_timestep_setters=()):
    """Flag stored step aliases/dynamic lookup and direct timestep mutations.

    This is a bounded audit of explicitly frozen files, not whole-program static
    proof. It does not discover aliases in unlisted code or native extensions.
    """
    setters = tuple(observed_timestep_setters)+((observed_timestep_setter,) if observed_timestep_setter is not None else ())
    files, findings = {}, []
    for path in sorted({str(Path(p).resolve()) for p in paths}):
        # Python source uses PEP 263/UTF-8, independent of a native library's
        # process locale. Native initialization may leave LC_CTYPE at C.
        with tokenize.open(path) as stream:
            source = stream.read()
        files[path] = file_hash(path)
        tree = ast.parse(source, filename=path)
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == 'step':
                parent = parents.get(node)
                if not (isinstance(parent, ast.Call) and parent.func is node):
                    findings.append(dict(path=path, line=node.lineno, reason='step_reference_not_direct_call'))
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ('getattr', 'setattr') and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == 'step'):
                findings.append(dict(path=path, line=node.lineno, reason='dynamic_step_lookup_or_write'))
            if (isinstance(node, ast.Attribute) and node.attr == 'timestep'
                    and isinstance(node.ctx, ast.Store)):
                ancestor = parents.get(node)
                while ancestor is not None and not isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ancestor = parents.get(ancestor)
                observed = ancestor is not None and any(
                    path == str(Path(code.co_filename).resolve())
                    and min([ancestor.lineno]+[n.lineno for n in ancestor.decorator_list]) == code.co_firstlineno
                    and ancestor.name == code.co_name for code in setters)
                if not observed:
                    findings.append(dict(path=path, line=node.lineno, reason='direct_timestep_write'))
    return dict(files=files, findings=findings, passed=bool(files) and not findings,
                limitation='Only listed frozen Python source; cached raw methods or calls from unlisted/native code are not automatically detectable.')


def _python_code(method):
    return getattr(getattr(method, '__func__', method), '__code__', None)


def timestep_setter_codes(scene_type):
    """Inspect class descriptors only, without evaluating a physical getter."""
    codes = {_python_code(getattr(scene_type, 'set_timestep', None))}
    descriptor = next((vars(cls)['timestep'] for cls in scene_type.__mro__ if 'timestep' in vars(cls)), None)
    if isinstance(descriptor, property):
        codes.add(_python_code(descriptor.fset))
    return tuple(code for code in codes if code is not None)


class ControlTimingObserver:
    """Count successful target backend calls and observe original CPU arguments."""
    def __init__(self, native, sample_function, *, phase, source_paths, journal_path=None):
        if phase not in ('expert_capture', 'legacy_eef_replay'):
            raise ValueError('Unsupported timing phase')
        self.native, self.scene = native, native.scene
        self.backend = self.scene.physx_system
        self.phase = phase
        self.dt_setter_codes = timestep_setter_codes(type(self.scene))
        self.scope = source_scope(source_paths, observed_timestep_setters=self.dt_setter_codes)
        self.backend_owner = next(cls for cls in type(self.backend).__mro__ if 'step' in vars(cls))
        self.original_step = vars(self.backend_owner)['step']
        try:
            own = vars(self.backend)
        except TypeError:
            own = {}
        if 'step' in own:
            raise ValueError('Instance step override prevents audited declaring-class observation')
        self.dt_before = float(self.scene.get_timestep())
        if not np.isfinite(self.dt_before) or self.dt_before <= 0:
            raise ValueError('Invalid runtime physics timestep')
        self.dt_after = None
        self.owner_thread = threading.get_ident()
        self.initial_other_threads = [t.ident for t in threading.enumerate() if t.ident != self.owner_thread]
        self.events, self.samples, self.errors = [], [], []
        self.steps_started = self.steps_completed = self.steps_failed = 0
        self.scene_calls = self.direct_backend_calls = self.command_returns = 0
        self.active_calls, self.scene_depth = {}, {}
        self.control_arguments_complete = True
        self.started = self.closed = False
        self.callback_seconds = 0.
        self.previous_profile = self.previous_thread_profile = None
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.journal = None
        self.identities = {}
        self.sample_code = _python_code(sample_function)
        if self.sample_code is None:
            raise ValueError('Sample function must have an inspectable Python code object')
        self.scene_code = _python_code(self.scene.step)
        if self.scene_code is None:
            raise ValueError('Scene wrapper must have an inspectable Python code object')
        for obj, names in ((native, ('delay', 'take_dense_action', 'take_action')),
                           (native.robot, ('set_arm_joints', 'set_gripper'))):
            for name in names:
                method = getattr(obj, name, None)
                if method is None:
                    continue
                code = _python_code(method)
                if code is None:
                    raise ValueError(f'Control method {name} is not observable Python')
                self.identities[(code, id(obj))] = name
        self.codes = {key[0] for key in self.identities} | {self.sample_code, self.scene_code}
        self.codes.update(self.dt_setter_codes)
        self.observed_code_sources = {str(Path(code.co_filename).resolve()): file_hash(code.co_filename)
                                      for code in self.codes if Path(code.co_filename).is_file()}
        if not set(self.observed_code_sources) <= set(self.scope['files']):
            self._error('observed_method_source_missing_from_audit_scope')

    def _error(self, value):
        if value not in self.errors:
            self.errors.append(value)

    def _emit(self, kind, **values):
        event = dict(sequence=len(self.events), phase=self.phase, kind=kind,
                     physics_step_completed=self.steps_completed,
                     simulated_seconds=self.steps_completed*self.dt_before,
                     wall_monotonic_ns=time.monotonic_ns(), thread=threading.get_ident())
        event.update(values)
        self.events.append(event)
        if self.journal is not None:
            self.journal.write(json.dumps(event, sort_keys=True, allow_nan=False)+'\n')
            if len(self.events) % 256 == 0 or kind in ('sample', 'control_return', 'backend_exception'):
                self.journal.flush()
        return event

    def _safe_emit(self, kind, **values):
        try:
            return self._emit(kind, **values)
        except Exception as exc:
            self._error(f'journal_or_observer_error:{type(exc).__name__}')
            return None

    def _backend_call(self, target, *args, **kwargs):
        if target is not self.backend:
            return self.original_step(target, *args, **kwargs)
        thread = threading.get_ident()
        if thread != self.owner_thread:
            self._error('target_backend_step_on_nonowner_thread')
        if not self.scene_depth.get(thread, 0):
            self.direct_backend_calls += 1
        self.steps_started += 1
        call_id = self.steps_started
        self._safe_emit('backend_begin', call_id=call_id,
                        via_scene_wrapper=bool(self.scene_depth.get(thread, 0)))
        try:
            result = self.original_step(target, *args, **kwargs)
        except BaseException as exc:
            self.steps_failed += 1
            self._error('backend_step_exception')
            self._safe_emit('backend_exception', call_id=call_id, exception_type=type(exc).__name__)
            raise
        else:
            self.steps_completed += 1
            self._safe_emit('backend_complete', call_id=call_id)
            return result

    def profile(self, frame, event, arg):
        if event not in ('call', 'return') or frame.f_code not in self.codes:
            return
        started = time.perf_counter()
        try:
            thread = threading.get_ident()
            obj = frame.f_locals.get('self')
            key = (thread, id(frame))
            if frame.f_code is self.scene_code and obj is self.scene:
                if event == 'call':
                    self.scene_calls += 1
                    self.scene_depth[thread] = self.scene_depth.get(thread, 0)+1
                    self.active_calls[key] = ('scene', self.steps_completed)
                else:
                    entry = self.active_calls.pop(key, None)
                    self.scene_depth[thread] = max(0, self.scene_depth.get(thread, 0)-1)
                    if entry is None or self.steps_completed-entry[1] != 1:
                        self._error('scene_return_without_exactly_one_completed_backend_step')
                return
            if frame.f_code in self.dt_setter_codes and obj is self.scene and event == 'call':
                self._error('timestep_setter_called_during_observation')
                return
            if frame.f_code is self.sample_code and frame.f_locals.get('native') is self.native:
                if event == 'return':
                    values, observed = cpu_value(arg)
                    if not observed:
                        self._error('sample_value_unobserved')
                    sample = dict(sample_index=len(self.samples),
                                  mapping='expert_original_picture_or_boundary' if self.phase == 'expert_capture' else 'legacy_actual_after_command',
                                  command_index=self.command_returns if self.phase == 'legacy_eef_replay' else None,
                                  command_index_semantics='Python take_action return-or-unwind count; verified against actual rows by the runner',
                                  physics_step_completed=self.steps_completed,
                                  simulated_seconds=self.steps_completed*self.dt_before,
                                  sample_sha256=json_hash(arg) if observed else None,
                                  cpu_value_sha256=json_hash(values), observed=observed)
                    self.samples.append(sample)
                    self._safe_emit('sample', **sample)
                return
            name = self.identities.get((frame.f_code, id(obj)))
            if name is None:
                return
            if thread != self.owner_thread:
                self._error('control_call_on_nonowner_thread')
            if event == 'call':
                names = frame.f_code.co_varnames[:frame.f_code.co_argcount+frame.f_code.co_kwonlyargcount]
                arguments = {name: frame.f_locals[name] for name in names if name != 'self'}
                values, observed = cpu_value(arguments)
                self.control_arguments_complete &= observed
                row = self._safe_emit('control_call', method=name, arguments=values,
                                      arguments_sha256=json_hash(values), arguments_observed=observed)
                self.active_calls[key] = (name, row['sequence'] if row else None)
            else:
                entry = self.active_calls.pop(key, None)
                if entry is None:
                    self._error('unpaired_control_return')
                if name == 'take_action':
                    self.command_returns += 1
                values, observed = cpu_value(arg)
                self._safe_emit('control_return', method=name, call_sequence=entry[1] if entry else None,
                                return_value_sha256=json_hash(values), return_observed=observed,
                                exception_status='not_inferred_from_python_return')
        except Exception as exc:
            self._error(f'profile_error:{type(exc).__name__}')
        finally:
            self.callback_seconds += time.perf_counter()-started

    def start(self):
        if self.started or self.closed:
            raise ValueError('Observer is one-shot')
        self.previous_profile = sys.getprofile()
        self.previous_thread_profile = threading.getprofile()
        if self.previous_profile is not None or self.previous_thread_profile is not None:
            raise RuntimeError('Existing profiler must not be replaced')
        if self.journal_path is not None:
            self.journal = self.journal_path.open('x', encoding='utf-8')
        observer = self
        def transparent_step(target, *args, **kwargs):
            return observer._backend_call(target, *args, **kwargs)
        self.wrapper = transparent_step
        try:
            setattr(self.backend_owner, 'step', self.wrapper)
            threading.setprofile(self.profile)
            sys.setprofile(self.profile)
            self.started = True
        except BaseException:
            sys.setprofile(self.previous_profile)
            threading.setprofile(self.previous_thread_profile)
            setattr(self.backend_owner, 'step', self.original_step)
            self._close_journal()
            raise
        return self

    def stop(self, *, exceptional=False):
        if self.closed:
            return
        if self.started:
            sys.setprofile(self.previous_profile)
            threading.setprofile(self.previous_thread_profile)
            if vars(self.backend_owner).get('step') is self.wrapper:
                try:
                    setattr(self.backend_owner, 'step', self.original_step)
                except Exception as exc:
                    self._error(f'backend_descriptor_restore_error:{type(exc).__name__}')
            else:
                self._error('backend_descriptor_changed_by_other_code_not_overwritten')
            try:
                self.dt_after = float(self.scene.get_timestep())
                if self.dt_after != self.dt_before:
                    self._error('runtime_timestep_changed')
            except Exception as exc:
                self._error(f'final_timestep_unreadable:{type(exc).__name__}')
        if exceptional:
            self._error('phase_exited_exceptionally')
        if self.active_calls:
            self._error('unfinished_profile_calls')
        self.closed = True
        self._close_journal()

    def _close_journal(self):
        # Observation I/O failures must never replace the native result/error.
        if self.journal is not None:
            try:
                self.journal.flush()
            except Exception as exc:
                self._error(f'journal_flush_error:{type(exc).__name__}')
            try:
                self.journal.close()
            except Exception as exc:
                self._error(f'journal_close_error:{type(exc).__name__}')

    def receipt(self):
        complete = bool(self.started and self.closed and self.scope['passed'] and not self.errors
                        and not self.initial_other_threads
                        and self.steps_started == self.steps_completed
                        and self.dt_after == self.dt_before)
        return dict(version=VERSION, phase=self.phase, clock_complete=complete,
                    control_arguments_complete=self.control_arguments_complete,
                    timing_compatibility_ready=complete and self.control_arguments_complete,
                    observed_code_sources=self.observed_code_sources,
                    backend_declaring_class=f'{self.backend_owner.__module__}.{self.backend_owner.__qualname__}',
                    completed_physics_steps=self.steps_completed, started_backend_steps=self.steps_started,
                    failed_backend_steps=self.steps_failed, direct_backend_calls=self.direct_backend_calls,
                    scene_wrapper_calls=self.scene_calls, simulated_seconds=self.steps_completed*self.dt_before,
                    runtime_dt_before=self.dt_before, runtime_dt_after=self.dt_after,
                    samples=self.samples, command_returns=self.command_returns,
                    command_returns_semantics='Python return-or-exception-unwind events; not an independent success counter',
                    source_scope=self.scope, initial_other_python_threads=self.initial_other_threads,
                    errors=self.errors, event_count=len(self.events), events_sha256=json_hash(self.events),
                    callback_seconds=self.callback_seconds,
                    callback_seconds_scope='Matched Python profile callbacks only; excludes dispatch/filter, backend shim and other I/O overhead. Not total instrumentation overhead.',
                    count_scope='Successful calls through the wrapped declaring-class System.step descriptor for this backend only. Cached raw bound methods/native bypasses outside audited Python sources are not detectable.',
                    timing_limit='Simulation dt is read before/after, with audited no-midphase-dt-write source; wall profiling/I/O overhead is not simulated time.',
                    formal_evidence_eligible=False, acceptance_ready=False, publication_allowed=False)

    def __enter__(self):
        return self.start()

    def __exit__(self, kind, value, traceback):
        self.stop(exceptional=kind is not None)
