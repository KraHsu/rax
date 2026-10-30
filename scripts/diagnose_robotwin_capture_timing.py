#!/usr/bin/env python3
"""Finite dev-only observation of existing expert capture and legacy EEF replay.

No controls are retimed. A failed capture/baseline is retained, never retried.
The separate diagnostic manifest is deliberately not an acceptance manifest.
"""
import argparse
import importlib
import json
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.robotwin.acceptance import derived_values_equal, trajectory_violations  # noqa: E402
from services.robotwin.acceptance_replay import (  # noqa: E402
    TRACKING, arrays, install_base_frame_adapter, replay, sample, tracking_report,
)
from services.robotwin.collection import atomic_json, initial_scene, scene_fingerprint  # noqa: E402
from services.robotwin.control_timing_observer import (  # noqa: E402
    ControlTimingObserver, file_hash, json_hash, source_scope, timestep_setter_codes,
)
from services.robotwin.programmatic_annotation import OutsideTemplate, template_hash  # noqa: E402
from services.robotwin.success_metadata import fresh_physical_state, initialize_success_metadata  # noqa: E402
from scripts import validate_robotwin_constraint_acceptance as legacy  # noqa: E402

KIND = 'capture_timing_stage_a_dev_only'
TASKS = ('dump_bin_bigbin', 'place_dual_shoes')
SEED = 10000
FLAGS = dict(formal_evidence_eligible=False, confidence_eligible=False, acceptance_ready=False,
             publication_allowed=False, training_allowed=False)
SOURCE_BASE = 'a2a4b356cac3ffe620389876b53652250bcd9635'


def read(path):
    return json.loads(Path(path).read_text())


def within(root, relative):
    result = (root/relative).resolve()
    if Path(relative).is_absolute() or not result.is_relative_to(root.resolve()) or result == root.resolve():
        raise ValueError('Diagnostic input must be a relative package file')
    return result


def validate_design(root, design, task):
    if (design.get('version') != 2 or design.get('kind') != KIND or design.get('split') != 'dev'
            or design.get('source_rax_base') != SOURCE_BASE or task not in TASKS
            or design.get('tasks') != list(TASKS)):
        raise ValueError('Wrong finite StageA task/design/base')
    for key, value in FLAGS.items():
        if design.get(key) is not value:
            raise ValueError('Diagnostic evidence cannot authorize acceptance or training')
    required = dict(seed=SEED, expert_attempts_per_task=1, legacy_replays_per_task=1,
                    candidate_variants=0, native_episodes_max=4, automatic_retries=0,
                    replace_failed_seed=False, continue_other_task_after_failure=True,
                    replay_requires_complete_clock=True, strict_tracking=TRACKING)
    for key, value in required.items():
        if design.get(key) != value:
            raise ValueError('Frozen finite budget/failure/tolerance contract changed: '+key)
    for relative, digest in design['input_files'].items():
        if file_hash(within(root, relative)) != digest:
            raise ValueError('Input changed: '+relative)
    for absolute, digest in design['external_files'].items():
        if not Path(absolute).is_absolute() or file_hash(absolute) != digest:
            raise ValueError('External source/config changed: '+absolute)
    for relative, digest in design['rax_files'].items():
        if file_hash(within(ROOT, relative)) != digest:
            raise ValueError('Execution source changed: '+relative)
    exclusions = read(within(root, design['training_exclusions']))
    if SEED in exclusions['excluded_training_seeds'] or exclusions['native_commit'] != design['native_commit']:
        raise ValueError('Training seed/native revision conflict')
    exposed = read(within(root, design['exposure_inventory']))
    if exposed.get('kind') != 'development_exposure_inventory' or not exposed.get('seed10000_declared_development_only'):
        raise ValueError('Existing development exposure must remain explicit')
    if SEED in exposed.get('reserved_holdout_seeds', []):
        raise ValueError('Reserved holdout seed cannot be used for StageA')
    templates = [legacy.load_template(within(root, p)) for p in design['templates'][task]]
    signatures = [tuple(s) for t in templates for s in t['signatures']]
    if (not templates or any(t['task'] != task for t in templates)
            or len(signatures) != len(set(signatures))):
        raise ValueError('Templates must route by unique original reference signature')
    expected = 2 if task == 'dump_bin_bigbin' else 1
    if len(templates) != expected:
        raise ValueError('All original published task branches must be sealed')
    paths = [str(within(ROOT, p)) for p in design['audit_rax_files']]+design['audit_native_files']
    minimum_rax = {'scripts/collect_robotwin_clean50.py', 'scripts/validate_robotwin_constraint_acceptance.py',
                   'services/robotwin/acceptance_replay.py', 'services/robotwin/collection.py'}
    native_root = Path(design['native_root'])
    minimum_native = {str(native_root/p) for p in ('envs/_base_task.py','envs/robot/robot.py',
                    'envs/robot/planner.py',*[f'envs/{t}.py' for t in TASKS])}
    if not minimum_rax <= set(design['audit_rax_files']) or not minimum_native <= set(design['audit_native_files']):
        raise ValueError('Mandatory native task/control/capture sources missing from audit')
    if not set(design['audit_rax_files']) <= set(design['rax_files']):
        raise ValueError('All audited RAX sources must be frozen')
    if not set(design['audit_native_files']) <= set(design['external_files']):
        raise ValueError('All audited external sources must be frozen')
    # Imports the wrapper class only; never creates a Scene/renderer/PhysX world.
    import sapien
    expected = design['runtime_numeric_context']
    if (sapien.__version__ != expected['sapien_version'] or np.__version__ != expected['numpy_version']
            or str(Path(sapien.pysapien.__file__).resolve()) != expected['backend_module']
            or file_hash(sapien.pysapien.__file__) != expected['backend_sha256']):
        raise ValueError('Native SAPIEN/backend/NumPy runtime differs from the CPU-verified context')
    scope = source_scope(paths, observed_timestep_setters=timestep_setter_codes(sapien.Scene))
    if not scope['passed']:
        raise ValueError('Unresolved step alias or timestep mutation in frozen scope: '+json.dumps(scope['findings']))
    return templates, exclusions, paths, scope


def routed_compiler(templates, original, selection):
    """Only pure label compilation varies; physical capture still runs once."""
    def compile_reference(_ignored, trajectory, seed, fingerprint):
        matches, rejections = [], []
        for template in templates:
            try:
                bundle, audit = original(template, trajectory, seed, fingerprint)
                matches.append((template, bundle, audit))
            except OutsideTemplate as exc:
                rejections.append(dict(template_sha256=template_hash(template), reason=str(exc)))
        selection['compilation_rejections'] = rejections
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one published template for observed reference; got {len(matches)}')
        template, bundle, audit = matches[0]
        selection.update(template=template, template_sha256=template_hash(template), signature=audit['signature'])
        return bundle, audit
    return compile_reference


def compare_sample_maps(reference, actual, expert, baseline):
    """Index equality is alignment, not equality of physical time or state."""
    def rows(trajectory):
        data = arrays(trajectory)
        return [{arm: dict(pose=data[arm]['pose'][i].tolist(), gripper=float(data[arm]['gripper'][i]))
                 for arm in ('left', 'right')} for i in range(len(data['left']['pose']))]
    source, measured = rows(reference), rows(actual)
    left, right = expert['samples'], baseline['samples']
    verified = (len(left) == len(source) and len(right) == len(measured)
                and all(m['sample_index'] == i and m['sample_sha256'] == json_hash(s)
                        for i, (m, s) in enumerate(zip(left, source)))
                and all(m['sample_index'] == i and m['command_index'] == i and m['sample_sha256'] == json_hash(s)
                        for i, (m, s) in enumerate(zip(right, measured))))
    pairs = [dict(sample_index=i, expert_step=l['physics_step_completed'], replay_step=r['physics_step_completed'],
                  expert_seconds=l['simulated_seconds'], replay_seconds=r['simulated_seconds'],
                  replay_minus_expert_seconds=r['simulated_seconds']-l['simulated_seconds'])
             for i, (l, r) in enumerate(zip(left, right))]
    return dict(original_sample_hashes_verified=verified, expert_rows=len(source), actual_rows=len(measured),
                duplicates_preserved=True, pair_alignment='same command index, NOT equal physics time', samples=pairs,
                timing_comparison_valid=verified and expert['clock_complete'] and baseline['clock_complete'],
                causal_failure_attribution=False)


def check_initial(native, reference, physical):
    if scene_fingerprint(initial_scene(native)) != reference['provenance']['scene_sha256']:
        raise ValueError('Replay scene differs from the captured scene')
    measured = sample(native)
    for arm in ('left', 'right'):
        expected = np.asarray(reference['trajectory'][arm]['pose'][0])
        actual = np.asarray(measured[arm]['pose'])
        if (not np.allclose(actual[:3], expected[:3], rtol=0, atol=1e-5)
                or abs(float(actual[3:] @ expected[3:])) < 1-1e-6
                or abs(measured[arm]['gripper']-reference['trajectory'][arm]['gripper'][0]) > 1e-5):
            raise ValueError('Replay initial EEF/gripper differs from reference')
    current = fresh_physical_state(native)
    if current != physical:
        raise ValueError('Exact initial physical state differs; timing comparison is blocked')
    return current


def run_task(output, task, templates, exclusions, source_paths, base_factory):
    """Injectable only for CPU tests; production factory is fixed below."""
    selection, capture_observers, physical = {}, [], []
    status = dict(task=task, seed=SEED, kind=KIND, status='capturing', native_episodes_started=0, **FLAGS)
    atomic_json(output/'run.json', status)

    def capture_factory(seed):
        status['native_episodes_started'] += 1
        atomic_json(output/'run.json', status)
        native = base_factory(seed)
        observer = None
        try:
            initial = fresh_physical_state(native)
            physical.append(initial)
            atomic_json(output/'expert_initial_physical_state.json', initial)
            observer = ControlTimingObserver(native, sample, phase='expert_capture', source_paths=source_paths,
                                             journal_path=output/'expert_events.jsonl')
            capture_observers.append(observer)
            original_close = native.close_env
            def close():
                native.close_env = original_close
                observer.stop()
                atomic_json(output/'expert_clock.json', observer.receipt())
                return original_close()
            native.close_env = close
            observer.start()
            return native
        except BaseException:
            if observer is not None:
                observer.stop(exceptional=True)
            native.close_env()
            raise

    original_compile = legacy.reference_bundle
    legacy.reference_bundle = routed_compiler(templates, original_compile, selection)
    try:
        reference = legacy.capture_reference(capture_factory, templates[0], SEED,
                        set(exclusions['excluded_training_scene_sha256']), attempt_path=output/'expert_attempt.json')
        atomic_json(output/'reference.json', reference)
        atomic_json(output/'template_selection.json', selection)
        expert = capture_observers[0].receipt()
        if not expert['timing_compatibility_ready']:
            raise ValueError('Expert observation scope/control arguments incomplete; no baseline launched')
    except BaseException as exc:
        status.update(status='blocked_capture', error=f'{type(exc).__name__}: {exc}')
        atomic_json(output/'template_selection.json', selection)
        atomic_json(output/'run.json', status)
        return status
    finally:
        legacy.reference_bundle = original_compile

    native, observer = None, None
    raw = dict(native_success=None, planner_success=False, execution_complete=False, realized=False)
    try:
        status.update(status='legacy_replay', native_episodes_started=2)
        atomic_json(output/'run.json', status)
        native = base_factory(SEED)
        initial = check_initial(native, reference, physical[0])
        atomic_json(output/'replay_initial_physical_state.json', initial)
        adapter = install_base_frame_adapter(native)
        if not derived_values_equal(adapter, reference['replay_adapter']):
            raise ValueError('Replay adapter differs from capture audit')
        observer = ControlTimingObserver(native, sample, phase='legacy_eef_replay', source_paths=source_paths,
                                         journal_path=output/'replay_events.jsonl')
        with observer:
            raw.update(replay(native, reference['trajectory']))
        if raw['execution_complete']:
            raw['tracking'] = tracking_report(raw['actual'], reference['trajectory'])
            raw['realized'] = raw['tracking']['realized']
            raw['strict_eligible'] = bool(raw['planner_success'] and raw['realized'])
            raw['strict_baseline_reliable'] = bool(raw['strict_eligible'] and raw['native_success'])
            raw['command_constraints'] = trajectory_violations(reference['bundle'], reference['trajectory'], reference['trajectory'])
            raw['actual_constraints'] = trajectory_violations(reference['bundle'], reference['trajectory'], raw['actual'])
        baseline = observer.receipt()
        if 'actual' in raw:
            comparison = compare_sample_maps(reference['trajectory'], raw['actual'], expert, baseline)
            atomic_json(output/'sample_time_comparison.json', comparison)
        else:
            comparison = {'timing_comparison_valid':False}
        status['status'] = ('diagnostic_complete' if raw.get('strict_baseline_reliable')
                            and baseline['timing_compatibility_ready'] and comparison['timing_comparison_valid']
                            else 'blocked_baseline')
    except BaseException as exc:
        raw.update(error=f'{type(exc).__name__}: {exc}')
        status.update(status='blocked_baseline', error=raw['error'])
    finally:
        if observer is not None:
            observer.stop()
            atomic_json(output/'replay_clock.json', observer.receipt())
        if native is not None:
            try:
                native.close_env()
            except BaseException as exc:
                raw['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                status['status'] = 'blocked_cleanup'
        atomic_json(output/'legacy_outcome.json', dict(raw, **FLAGS))
        atomic_json(output/'run.json', status)
    return status


def commit_index(output):
    """Evidence files are immutable; incomplete runs remain visibly incomplete."""
    files = {p.name:file_hash(p) for p in sorted(output.iterdir())
             if p.is_file() and p.suffix in ('.json','.jsonl') and p.name != 'artifact_index.json'}
    atomic_json(output/'artifact_index.json', dict(kind=KIND, files=files, **FLAGS))


def verify_output(output, expected_design_sha256):
    output = Path(output)
    index = read(output/'artifact_index.json')
    if index.get('kind') != KIND or any(index.get(k) is not v for k,v in FLAGS.items()):
        raise ValueError('Diagnostic index role changed')
    observed = {p.name for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl')
                and p.name != 'artifact_index.json'}
    if set(index['files']) != observed:
        raise ValueError('Unexpected or missing committed evidence files')
    for name,digest in index['files'].items():
        if file_hash(within(output,name)) != digest:
            raise ValueError('Committed diagnostic evidence changed: '+name)
    manifest, state = read(output/'diagnostic_manifest.json'), read(output/'run.json')
    if (manifest.get('kind') != KIND or manifest['design_sha256'] != expected_design_sha256
            or manifest['seed'] != SEED or manifest['task'] not in TASKS
            or state['task'] != manifest['task'] or state['seed'] != SEED
            or state['native_episodes_started'] not in (1,2)
            or any(manifest.get(k) is not v or state.get(k) is not v for k,v in FLAGS.items())):
        raise ValueError('Diagnostic manifest/state identity changed')
    # Recompute the complete committed journal rather than trusting receipt counters.
    clocks = {}
    for phase in ('expert','replay'):
        receipt = output/(phase+'_clock.json')
        if not receipt.exists():
            continue
        clocks[phase] = value = read(receipt)
        events = [json.loads(line) for line in (output/(phase+'_events.jsonl')).read_text().splitlines()]
        completed = [e for e in events if e['kind']=='backend_complete']
        if (value['events_sha256'] != json_hash(events) or value['event_count'] != len(events)
                or [e['sequence'] for e in events] != list(range(len(events)))
                or value['completed_physics_steps'] != len(completed)
                or [e['physics_step_completed'] for e in completed] != list(range(1,len(completed)+1))):
            raise ValueError('Clock journal does not recompute: '+phase)
        sampled = [e for e in events if e['kind']=='sample']
        if len(sampled) != len(value['samples']) or any(
                any(e.get(k) != v for k,v in s.items()) for e,s in zip(sampled,value['samples'])):
            raise ValueError('Clock samples changed: '+phase)
    if state['status']=='diagnostic_complete':
        if state['native_episodes_started'] != 2 or set(clocks) != {'expert','replay'}:
            raise ValueError('A complete diagnostic requires both fixed episodes')
        raw, reference = read(output/'legacy_outcome.json'), read(output/'reference.json')
        actual_tracking = tracking_report(raw['actual'],reference['trajectory'])
        comparison = compare_sample_maps(reference['trajectory'],raw['actual'],clocks['expert'],clocks['replay'])
        if (not raw['execution_complete'] or not raw['native_success'] or not raw['planner_success']
                or not actual_tracking['realized'] or not derived_values_equal(raw['tracking'],actual_tracking)
                or not comparison['timing_comparison_valid']
                or read(output/'sample_time_comparison.json') != comparison
                or any(not c['timing_compatibility_ready'] for c in clocks.values())):
            raise ValueError('Complete status contradicts retained baseline/clock evidence')
    return dict(kind=KIND,status=state['status'],native_episodes_started=state['native_episodes_started'],
                artifact_index_sha256=file_hash(output/'artifact_index.json'),**FLAGS)


def run(args):
    root = args.design.resolve().parent
    design = read(args.design)
    templates, exclusions, sources, scope = validate_design(root, design, args.task)
    from services.robotwin.runtime import initialize_runtime, native_config, source_commit
    if source_commit() != design['native_commit'] or legacy.training_exclusions(Path(design['training_root'])) != exclusions:
        raise ValueError('Native revision or training exclusion inventory changed')
    if args.check_only:
        print(json.dumps(dict(task=args.task, checked=True, native_episodes_max=2, scope=scope,
                              design_sha256=file_hash(args.design), gpu_started=False, **FLAGS)))
        return 0
    if not args.execute:
        raise ValueError('Explicit execute flag required; default is no launch')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'diagnostic_manifest.json', dict(kind=KIND, version=2, split='dev', task=args.task,
                seed=SEED, design=design, design_sha256=file_hash(args.design), created_unix=time.time(), **FLAGS))
    def interrupted(signum, _frame):
        raise InterruptedError(f'Finite diagnostic interrupted by signal {signum}')
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupted)
    initialize_runtime()
    from services.robotwin.collection import install_planner_cache
    install_planner_cache()
    native_class = getattr(importlib.import_module('envs.'+args.task), args.task)
    kwargs = native_config(args.task, 0)
    kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                  eval_video_save_dir=None, save_path=str(output/'native_scratch'))
    def factory(seed):
        random.seed(seed)
        native = native_class()
        try:
            native.setup_demo(now_ep_num=0, seed=seed, is_test=False, **kwargs)
            initialize_success_metadata(args.task, native,
                                        expected_scene_sha256=scene_fingerprint(initial_scene(native)))
        except BaseException:
            native.close_env()
            raise
        return native
    result = run_task(output, args.task, templates, exclusions, sources, factory)
    commit_index(output)
    verify_output(output,file_hash(args.design))
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'diagnostic_complete' else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--execute', action='store_true')
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
