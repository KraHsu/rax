"""No Scene, CUDA or native factory: source-predicate and frozen-candidate probes."""
import ast
import gzip
import json
from pathlib import Path
import types

import numpy as np

from candidate_design import generate
from contracts import digest, file_hash, read
from predicate_observer import PredicateObserver, verify_observations

PACKAGE = Path(__file__).resolve().parents[1]


def compare(a, b, path='', differences=None):
    differences = [] if differences is None else differences
    if isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError('CPU pilot keys differ: '+path)
        for k in a:
            compare(a[k], b[k], path+'/'+k, differences)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            raise ValueError('CPU pilot list differs: '+path)
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, path+'/'+str(i), differences)
    elif type(a) is float and type(b) in (float, int):
        if a != b:
            difference = abs(a-b)
            if difference > 1e-12 or not np.isfinite(difference):
                raise ValueError('CPU pilot numerical change exceeds declared comparison tolerance: '+path)
            differences.append(dict(path=path, old=a, new=b, absolute_difference=difference))
    elif a != b:
        raise ValueError('CPU pilot nonnumeric selection/evidence differs: '+path)
    return differences


def run():
    design = read(PACKAGE/'design/runtime_design.json')
    if np.__version__ != design['numeric_context']['numpy_version']:
        raise ValueError('Run this preflight in the frozen native NumPy environment')
    rows = []
    refs = json.loads(gzip.decompress((PACKAGE/'design/inputs/development_reference_pilot.json.gz').read_bytes()))
    for seed, ref in sorted(refs.items()):
        result = json.loads(json.dumps(generate(ref['trajectory'], int(seed), ref['provenance']['scene_sha256'])))
        expected = json.loads(gzip.decompress((PACKAGE/f'design/v3/old_reference_pilot/seed{seed}.json.gz').read_bytes()))
        expected['generator_sha256'] = result['generator_sha256']
        pilot_numpy = expected['numpy_version']
        expected['numpy_version'] = result['numpy_version']
        # Derived hashes reflect possible CPU NumPy rounding changes; compare
        # every underlying numeric/semantic field instead of pretending hashes
        # are unchanged. The new runtime generation is always fully frozen.
        hash_differences = []
        def normalized(value, target, path=''):
            if isinstance(value, dict):
                for k, v in value.items():
                    if k in {'raw_trajectory_sha256', 'selected_trajectory_sha256', 'trajectory_sha256'} and target[k] != v:
                        hash_differences.append(dict(path=path+'/'+k, old=v, new=target[k]))
                        value[k] = target[k]
                    else:
                        normalized(v, target[k], path+'/'+k)
            elif isinstance(value, list):
                for i, (v, t) in enumerate(zip(value, target)):
                    normalized(v, t, path+'/'+str(i))
        normalized(expected, result)
        numeric_differences = compare(expected, result)
        rows.append(dict(seed=int(seed), slots=len(result['slots']),
            scores=sum(len(s['attempts']) for s in result['slots']),
            generation_sha256=digest(result), pilot_numpy_version=pilot_numpy, numeric_differences=numeric_differences,
            inside=sum(s['commanded_domain']=='inside' for s in result['slots']),
            outside=sum(s['commanded_domain']=='outside' for s in result['slots']),
            all_raw_slots_retained=len(result['slots'])==18,
            derived_hash_differences=hash_differences, unchanged_selection_and_pass_flags=True))
    source = Path(design['native_root'])/'envs/pick_dual_bottles.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    fn = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == 'check_success')
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source.resolve()), 'exec'), ns)
    class Actor:
        def __init__(self, value):
            self.value, self.calls = value, 0
        def get_functional_point(self, index):
            assert index == 0
            self.calls += 1
            return self.value.copy()
    native = types.SimpleNamespace(bottle1=Actor(np.array([-.06, -.105, 1.], dtype=np.float32)),
        bottle2=Actor(np.array([.06, -.105, 1.], dtype=np.float32)),
        left_target_pose=[-.06, -.105, 1.], right_target_pose=[.06, -.105, 1.],
        actor_pose=True, take_action_cnt=0)
    native.check_success = types.MethodType(ns['check_success'], native)
    monitor = PredicateObserver(native, source)
    monitor.install()
    try:
        outcomes = [bool(native.check_success())]
        native.bottle1.value[2] = .77
        outcomes.append(bool(native.check_success()))
    finally:
        monitor.close()
    assert outcomes == [True, False] and native.actor_pose is False
    assert native.bottle1.calls == native.bottle2.calls == 2
    observation = monitor.result()
    verify_observations(observation)
    report = dict(status='CPU_only_no_Scene_no_native_factory_no_GPU', numpy_version=np.__version__,
        design_sha256=file_hash(PACKAGE/'design/runtime_design.json'),
        predicate_source_sha256=file_hash(source), predicate_source_ast_sha256=digest(ast.dump(fn, include_attributes=False)),
        source_predicate_returns=outcomes, original_getter_calls_each=2, extra_predicate_calls=0,
        actor_pose_side_effect_preserved=True, observation_complete=observation['observation_complete'],
        candidate_rows=rows, candidate_slots=sum(r['slots'] for r in rows),
        scores=sum(r['scores'] for r in rows), native_initializations=0,
        max_numeric_drift=max([d['absolute_difference'] for r in rows for d in r['numeric_differences']] or [0]),
        scope='Only already exposed old3references; no new seed expert or physical outcome has run.')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    run()
