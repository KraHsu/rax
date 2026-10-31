"""Finite three-scene development driver. No resume, reseed or native success selection."""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'code/rax'))
sys.path.insert(0, str(PACKAGE/'code/owner_dependencies'))
from scripts.robotwin_dev_queue import process_identity
from contracts import (SEEDS, TASK, assert_seed_exclusion, claim_scene, digest, file_hash,
                       exposure_scope, inspect_registry, new_development_ledger, read, training_identity,
                       update_state, write_once)
from candidate_design import fixed_reference, generate
from predicate_observer import PredicateObserver, verify_observations
from scripts.search_robotwin_constraint_counterexamples import check_initial, classify_trial
from scripts.validate_robotwin_constraint_acceptance import reference_bundle
from services.robotwin.acceptance import annotation_policy_fingerprint
from services.robotwin.acceptance_replay import (ALIGNMENT, SUCCESS_RULE, install_base_frame_adapter,
                                               replay, sample, samples_to_trajectory, tracking_report)
from services.robotwin.collection import initial_scene, scene_fingerprint


class ScientificStop(Exception):
    pass


def contained(relative):
    path = (PACKAGE/relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(PACKAGE) or path == PACKAGE:
        raise ValueError('Contained package-relative path required')
    return path


def validate_design(design, *, runtime=True):
    if (design['kind'] != 'dual_bottle_finite_development_v2' or design['task'] != TASK
            or design['seeds'] != list(SEEDS) or design['maximum_native_episodes'] != 63
            or design['automatic_retries'] != 0 or design['max_captures_per_scene'] != 1
            or design['max_baselines_per_scene'] != 2 or design['slots_per_scene'] != 18
            or design['fixed_candidate_count'] != 54 or design['maximum_baselines'] != 6
            or design['maximum_experts'] != 3 or design['schema_version'] != '1.5'
            or design['generation_policy'] != 'all18_raw_once_no_projection_no_feedback'
            or design['template'] != 'design/v3/template.json'
            or design['training_or_publication_allowed'] is not False):
        raise ValueError('Finite task/seeds/budget changed')
    for relative, expected in design['frozen_inputs'].items():
        if file_hash(contained(relative)) != expected:
            raise ValueError('Frozen input changed: '+relative)
    for absolute, expected in design['external_files'].items():
        if file_hash(absolute) != expected:
            raise ValueError('External native/data/source changed: '+absolute)
    training = training_identity(design['training_root'])
    frozen_training = read(contained(design['training_identity']))
    if training != frozen_training:
        raise ValueError('Dataset version/provenance/numeric labels changed')
    registry_rows = [inspect_registry(row['root']) for row in design['registry_inspection']]
    # Each explicit known path is inspected, including a frozen absence. A new
    # registry requires review instead of silently inheriting a guessed authority.
    if registry_rows != read(contained(design['registry_snapshot'])):
        raise ValueError('Exposure registry state changed')
    frozen_exposure = read(contained(design['exposure_inventory']))
    exposure = exposure_scope(design['exposure_scan_roots'], ignore_roots=[PACKAGE])
    if ({row['root'] for row in exposure['explicit_registry_fields']}
            != {row['root'] for row in frozen_exposure['explicit_registry_fields']}
            or exposure['configured_registry_snapshots'] != frozen_exposure['configured_registry_snapshots']):
        raise ValueError('Declared registry scope/state changed; review required')
    exposure['frozen_inventory_sha256'] = file_hash(contained(design['exposure_inventory']))
    exposure['excluded_seeds'] = sorted(set(exposure['excluded_seeds']) | set(frozen_exposure['excluded_seeds']))
    exposure['excluded_scene_sha256'] = sorted(set(exposure['excluded_scene_sha256']) | set(frozen_exposure['excluded_scene_sha256']))
    assert_seed_exclusion(design['seeds'], training, registry_rows, exposure['excluded_seeds'])
    if runtime and np.__version__ != design['numeric_context']['numpy_version']:
        raise ValueError('Generator NumPy runtime differs from frozen native CPU environment')
    return training, registry_rows, exposure


def record_artifact(root, relative, value):
    path = root/relative
    write_once(path, value)
    return dict(path=relative, sha256=file_hash(path))


def checked_artifact(root, spec):
    path = (root/spec['path']).resolve()
    if not path.is_relative_to(root.resolve()) or file_hash(path) != spec['sha256']:
        raise ValueError('Evidence artifact changed or escaped run')
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == '.gz' else read(path)


def save_observations(path, observer):
    value = observer.result()
    # Diagnostics can be incomplete without changing the original native result.
    verify_observations(value)
    encoded = gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
    with Path(path).open('xb') as handle:
        handle.write(encoded)
    return value


def capture(factory, template, seed, excluded, ledger, root, design):
    """Original capture control path, with explicit scientific-vs-infrastructure failures."""
    from scripts.collect_robotwin_clean50 import play_expert, RejectedEpisode
    native, observer = None, None
    samples, ref, failure, fatal = [], None, None, None
    attempt = dict(seed=seed, task=TASK, native_initializations_started=1,
                   expert_attempts_started=0, native_success=None, planner_success=None,
                   native_actions_executed=False, status='started', automatic_retries=0)
    write_once(root/'capture_intent.json', copy.deepcopy(attempt))
    started = time.monotonic()
    try:
        native = factory(seed)
        scene = initial_scene(native)
        fingerprint = scene_fingerprint(scene)
        attempt.update(initial_scene=scene, scene_sha256=fingerprint)
        adapter = install_base_frame_adapter(native, install=False)
        try:
            claim = claim_scene(ledger, seed, fingerprint, excluded)
        except ValueError as exc:
            # A fresh-scene duplicate is a predeclared unsupported scene, not a
            # reason to secretly select another seed. Other ledger faults are fatal.
            if str(exc).startswith('Fresh scene duplicates'):
                raise ScientificStop(str(exc)) from exc
            raise
        attempt['development_ledger_after_claim_sha256'] = claim
        observer = PredicateObserver(native, Path(design['native_root'])/'envs'/f'{TASK}.py')
        observer.phase = 'expert_capture'
        observer.install()
        def observe_sample(*_args, **_kwargs):
            samples.append(sample(native))
            observer.observe_pose('reference_save_freq_sample', len(samples)-1)
        observe_sample()
        native._take_picture = observe_sample
        attempt.update(native_actions_executed=True, expert_attempts_started=1)
        play_expert(native)
        observe_sample()
        attempt['planner_success'] = bool(native.plan_success)
        observer.phase = 'original_capture_final_check'
        attempt['native_success'] = bool(native.check_success()) if native.plan_success else None
        if not attempt['planner_success'] or not attempt['native_success']:
            raise ScientificStop('reference_native_expert_failed')
        trajectory = samples_to_trajectory(samples)
        try:
            fixed_reference(trajectory)
        except ValueError as exc:
            if str(exc).startswith('Unsupported'):
                raise ScientificStop(str(exc)) from exc
            raise
        bundle, audit = reference_bundle(template, trajectory, seed, fingerprint)
        ref = dict(trajectory=trajectory, bundle=bundle, template_audit=audit, initial_scene=scene,
                   replay_adapter=adapter, replay_adapter_applied=False,
                   provenance=dict(task=TASK, seed=seed, scene_sha256=fingerprint, mode='demo_clean',
                       split='validation', source_episode_sha256=digest(dict(scene=scene, trajectory=trajectory)),
                       recording_method='native_expert_save_freq_trace_no_images'))
        attempt['status'] = 'captured'
    except ScientificStop as exc:
        failure = str(exc)
        attempt.update(status='blocked_reference', stop_reason=failure)
    except RejectedEpisode as exc:
        failure = 'native_expert_rejected: '+str(exc)
        attempt.update(status='blocked_reference', stop_reason=failure,
                       planner_success=bool(getattr(native, 'plan_success', False)),
                       rejected_episode=str(exc))
    except BaseException as exc:
        fatal = exc
        attempt.update(status='infrastructure_failed', error=type(exc).__name__+': '+str(exc))
    finally:
        if observer is not None:
            try:
                observer.close()
            except BaseException as exc:
                fatal = fatal or exc
                attempt.update(status='infrastructure_failed', observer_close_error=type(exc).__name__+': '+str(exc))
        try:
            if native is not None:
                native.close_env()
        except BaseException as exc:
            fatal = fatal or exc
            attempt.update(status='infrastructure_failed', cleanup_error=type(exc).__name__+': '+str(exc))
        if observer is not None:
            try:
                save_observations(root/'capture_observations.json.gz', observer)
            except BaseException as exc:
                fatal = fatal or exc
                attempt.update(status='infrastructure_failed', observation_persistence_error=type(exc).__name__+': '+str(exc))
        attempt.update(seconds=time.monotonic()-started, samples=samples)
        write_once(root/'capture_attempt.json', attempt)
    if fatal is not None:
        raise RuntimeError('Capture infrastructure failure; no next scene/retry') from fatal
    return ref, failure


def execute_trial(factory, reference, candidate, seed, trial_id, root, design, baseline=None):
    trial = root/'trials'/trial_id
    trial.mkdir(parents=True, exist_ok=False)
    write_once(trial/'intent.json', dict(seed=seed, trial_id=trial_id, commanded_sha256=digest(candidate['trajectory']),
                                       native_initializations_started=1, automatic_retries=0))
    raw = dict(native_success=None, planner_success=False, execution_complete=False,
               realized=False, ever_success=False, final_success=False,
               alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE)
    actual = {'left': {'pose': [], 'gripper': []}, 'right': {'pose': [], 'gripper': []}}
    native, observer, fatal, original_take = None, None, None, None
    started = time.monotonic()
    try:
        native = factory(seed)
        if scene_fingerprint(initial_scene(native)) != reference['provenance']['scene_sha256']:
            raise ValueError('Fresh replay scene differs from captured reference')
        adapter = install_base_frame_adapter(native)
        if adapter != reference['replay_adapter']:
            raise ValueError('Replay assets/adapter changed')
        measured = sample(native)
        first = {a: {'pose': [measured[a]['pose']]*len(reference['trajectory'][a]['pose']),
                     'gripper': [measured[a]['gripper']]*len(reference['trajectory'][a]['gripper'])} for a in ('left', 'right')}
        check_initial(reference['trajectory'], first, physical=True)
        raw.update(replay_adapter=adapter, replay_adapter_applied=True)
        observer = PredicateObserver(native, Path(design['native_root'])/'envs'/f'{TASK}.py')
        observer.phase = 'replay_existing_predicate_cadence'
        observer.install()
        observer.observe_pose('initial', 0)
        original_take = native.take_action
        def observed_take(*args, **kwargs):
            observer.command_frame = int(native.take_action_cnt)+1
            result = original_take(*args, **kwargs)
            observer.observe_pose('command_end', int(native.take_action_cnt))
            return result
        native.take_action = observed_take
        raw.update(replay(native, candidate['trajectory']))
        actual = raw.pop('actual')
        if raw.get('error'):
            # Original replay returns an incomplete trace on an exception. Keep
            # it, but do not assume this is an ordinary task failure and continue.
            fatal = RuntimeError(raw['error'])
            raw['execution_failure_class'] = 'controller_execution_exception_campaign_stops'
        elif raw['execution_complete']:
            track = tracking_report(actual, candidate['trajectory'], baseline=baseline,
                                    reference=reference['trajectory'] if baseline is not None else None)
            raw.update(realized=track['realized'], tracking=track,
                       execution_failure_class=None if raw['planner_success'] and track['realized'] else 'controller_planner_or_tracking_invalid')
    except BaseException as exc:
        fatal = exc
        raw.update(error=type(exc).__name__+': '+str(exc), execution_complete=False,
                   execution_failure_class='factory_adapter_or_controller_infrastructure_exception')
    finally:
        if observer is not None:
            try:
                observer.close()
            except BaseException as exc:
                fatal = fatal or exc
                raw['observer_close_error'] = type(exc).__name__+': '+str(exc)
        if native is not None:
            if original_take is not None:
                try:
                    native.take_action = original_take
                except BaseException as exc:
                    fatal = fatal or exc
                    raw['take_action_restore_error'] = type(exc).__name__+': '+str(exc)
            try:
                native.close_env()
            except BaseException as exc:
                fatal = fatal or exc
                raw.update(cleanup_error=type(exc).__name__+': '+str(exc),
                           execution_failure_class='cleanup_infrastructure_failure')
    raw.update(trial_id=trial_id, scene_sha256=reference['provenance']['scene_sha256'], seconds=time.monotonic()-started)
    trace_spec = record_artifact(root, f'trials/{trial_id}/trajectory.json', dict(commanded=candidate['trajectory'], actual=actual))
    obs_spec = None
    if observer is not None:
        try:
            obs = save_observations(trial/'observations.json.gz', observer)
            obs_spec = dict(path=f'trials/{trial_id}/observations.json.gz', sha256=file_hash(trial/'observations.json.gz'))
            if obs['observation_complete'] and raw['execution_complete']:
                if verify_observations(obs)['any_recorded_success'] is not raw['native_success']:
                    raise ValueError('Observed original predicate results disagree with native replay outcome')
        except BaseException as exc:
            fatal = fatal or exc
            raw['observation_persistence_error'] = type(exc).__name__+': '+str(exc)
    outcome_spec = record_artifact(root, f'trials/{trial_id}/outcome.json', raw)
    result = classify_trial(reference, candidate, raw, actual, baseline=baseline)
    record = dict(trial_id=trial_id, seed=seed, reference_sha256=digest(reference),
                  candidate_sha256=digest(candidate), trajectory=trace_spec, outcome=outcome_spec,
                  observations=obs_spec, classification=result,
                  infrastructure_failure=fatal is not None,
                  infrastructure_error=None if fatal is None else type(fatal).__name__+': '+str(fatal),
                  automatic_retries=0)
    write_once(trial/'record.json', record)
    if fatal is not None:
        raise RuntimeError('Execution infrastructure failure preserved; campaign stops') from fatal
    return result, actual


def make_factory(design, root):
    from services.robotwin.runtime import initialize_runtime
    from services.robotwin.collection import install_planner_cache
    from scripts.collect_robotwin_clean50 import native_config
    initialize_runtime()
    install_planner_cache()
    native_class = getattr(importlib.import_module('envs.'+TASK), TASK)
    kwargs = native_config(TASK, 0)
    kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                  eval_video_save_dir=None, save_path=str(root/'native_scratch'))
    def factory(seed):
        random.seed(seed)
        native = native_class()
        try:
            native.setup_demo(now_ep_num=0, seed=seed, is_test=False, **kwargs)
        except BaseException as primary:
            try:
                native.close_env()
            except BaseException as cleanup:
                raise RuntimeError('Factory setup failed: '+type(primary).__name__+': '+str(primary)
                                   +'; cleanup also failed: '+type(cleanup).__name__+': '+str(cleanup)) from primary
            raise
        return native
    return factory


def selected_candidates(generation):
    if (generation.get('version') != 2
            or generation.get('selection_policy') != 'all18_raw_once_no_projection_no_feedback'
            or [s['spec']['slot'] for s in generation['slots']] != list(range(18))):
        raise ValueError('Exactly18 ordered fixed raw slots required')
    result = []
    for slot in generation['slots']:
        if (len(slot['attempts']) != 1 or slot['selected_alpha'] != 1.0
                or slot['attempts'][0]['alpha'] != 1.0
                or slot['raw_trajectory'] != slot['attempts'][0]['trajectory']
                or slot['raw_trajectory_sha256'] != digest(slot['raw_trajectory'])
                or slot['selected_trajectory_sha256'] != slot['raw_trajectory_sha256']
                or slot['planned_native_eligible'] is not True):
            raise ValueError('Raw candidate was projected, omitted, or replaced')
        selected = slot['attempts'][0]
        inside = bool(selected['score']['commanded_constraint_pass'])
        if slot['commanded_domain'] != ('inside' if inside else 'outside'):
            raise ValueError('Frozen commanded domain differs from score')
        result.append(dict(candidate_id=slot['spec']['candidate_id'], family=slot['spec']['family'],
            source='crafted', slot=slot['spec']['slot'], category=slot['spec']['category'],
            commanded_domain=slot['commanded_domain'], declared_outside_control=slot['declared_outside_control'],
            selected_alpha=1.0, trajectory=slot['raw_trajectory']))
    return result


def frozen_manifest(root, design, seed, generation):
    return dict(version=2, task=TASK, seed=seed, reference_sha256=file_hash(root/'reference.json'),
        generation_sha256=file_hash(root/'generation.json.gz'), design_sha256=design['identity_sha256'],
        annotation_policy_sha256=annotation_policy_fingerprint(), candidates=selected_candidates(generation),
        baseline_ids=['baseline_0', 'baseline_1'],
        unavailable_slots=[], candidate_construction='all18_raw_once_no_projection_no_feedback',
        candidate_scores_frozen_before_baseline=True,
        predicate_policy='native_ever_success_original_cadence_existing_fixedN_replay_protocol',
        alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE, early_termination_suppressed=True,
        dataset_version_sha256=design['dataset_version_sha256'], source='crafted_not_model',
        publication_allowed=False, training_allowed=False, formal_holdout=False)


def run_scene(design, seed, output, ledger, exclusions, *, factory=None, preflight=None):
    if seed not in SEEDS or output.exists():
        raise ValueError('Only a new fixed scene output is allowed')
    output.mkdir(parents=True)
    write_once(output/'preflight.json', dict(excluded_scene_sha256=sorted(exclusions),
        dataset_version_sha256=design['dataset_version_sha256'], details=preflight))
    state = dict(seed=seed, status='capturing_reference', dataset_version_sha256=design['dataset_version_sha256'],
                 automatic_retries=0, native_initializations_max=21)
    update_state(output/'state.json', state)
    try:
        factory = factory or make_factory(design, output)
        template = read(PACKAGE/'design/v3/template.json')
        reference, stop = capture(factory, template, seed, exclusions, ledger, output, design)
        if stop is not None:
            state.update(status='blocked_reference', reason=stop)
            update_state(output/'state.json', state)
            return state
        write_once(output/'reference.json', reference)
        generation = generate(reference['trajectory'], seed, reference['provenance']['scene_sha256'])
        if generation['bundle'] != reference['bundle']:
            raise ValueError('Candidate generation changed fixed-reference labels')
        encoded = gzip.compress(json.dumps(generation, separators=(',', ':'), allow_nan=False).encode(), mtime=0)
        with (output/'generation.json.gz').open('xb') as handle:
            handle.write(encoded)
        candidates = selected_candidates(generation)
        # Every command is frozen before baseline0, not chosen from native feedback.
        manifest = frozen_manifest(output, design, seed, generation)
        write_once(output/'manifest.json', manifest)
        state.update(status='running_baselines', scene_sha256=reference['provenance']['scene_sha256'])
        update_state(output/'state.json', state)
        baseline = None
        for i in range(2):
            candidate = dict(source='expert', family='baseline', trajectory=reference['trajectory'])
            result, actual = execute_trial(factory, reference, candidate, seed, f'baseline_{i}', output, design, baseline)
            if not (result['eligible'] and result['native_success'] and result['commanded_constraint_pass'] and result['actual_constraint_pass']):
                state.update(status='blocked_baseline', blocked_trial=f'baseline_{i}')
                update_state(output/'state.json', state)
                return state
            if i == 0:
                baseline = actual
        state['status'] = 'running_predeclared_candidates'
        update_state(output/'state.json', state)
        for candidate in candidates:
            execute_trial(factory, reference, candidate, seed, f"candidate_{candidate['slot']:02d}", output, design, baseline)
        state.update(status='scene_complete', selected_candidates=len(candidates), unavailable_slots=manifest['unavailable_slots'])
        update_state(output/'state.json', state)
        return state
    except BaseException as exc:
        state.update(status='infrastructure_failed', error=type(exc).__name__+': '+str(exc))
        update_state(output/'state.json', state)
        raise


def run(args):
    design = read(args.design)
    training, registry_rows, exposure = validate_design(design)
    design['identity_sha256'] = file_hash(args.design)
    if args.check_only:
        print(json.dumps(dict(check='passed', task=TASK, gpu_started=False, maximum_native_episodes=63,
                              dataset_version_sha256=training['dataset_version_sha256'])))
        return 0
    if not args.execute:
        raise ValueError('Explicit finite execution flag required')
    campaign_output = contained(design['output'])
    if args.seed is not None:
        if args.seed not in SEEDS or args.ledger is None:
            raise ValueError('Worker needs exactly one fixed seed and coordinator ledger')
        if (args.output.resolve() != campaign_output/'scenes'/f'seed{args.seed}'
                or args.ledger.resolve() != campaign_output/'development_registry'/f'{TASK}.json'
                or read(args.ledger)['plan_sha256'] != design['identity_sha256']):
            raise ValueError('Worker output/registry belongs to another finite campaign')
        worker_intent = read(campaign_output/f'worker_intent_{args.seed}.json')
        parent_identity = process_identity(os.getppid())
        own_identity = process_identity(os.getpid())
        if (parent_identity is None or own_identity is None
                or worker_intent['coordinator'] != parent_identity
                or worker_intent['seed'] != args.seed):
            raise ValueError('Internal scene worker lacks its exact live coordinator parent')
        write_once(campaign_output/'worker_receipts'/f'seed{args.seed}.json',
                   dict(seed=args.seed, coordinator=parent_identity, worker=own_identity,
                        design_sha256=design['identity_sha256'], automatic_retries=0))
        prior = set(training['exclusions']['excluded_training_scene_sha256']) | set(exposure['excluded_scene_sha256'])
        for row in registry_rows:
            prior.update(row['excluded_scenes'])
        run_scene(design, args.seed, args.output, args.ledger, prior,
                  preflight=dict(registries=registry_rows, exposure=exposure,
                                 training_identity_sha256=digest(training)))
        return 0
    if args.output.resolve() != campaign_output:
        raise ValueError('Coordinator output must match frozen package plan')
    args.output.mkdir(parents=True, exist_ok=False)
    coordinator_identity = process_identity(os.getpid())
    if coordinator_identity is None:
        raise ValueError('Coordinator process identity is unavailable')
    ledger = args.output/'development_registry'/f'{TASK}.json'
    write_once(ledger, new_development_ledger(design['identity_sha256']))
    write_once(args.output/'intent.json', dict(design_sha256=design['identity_sha256'], seeds=list(SEEDS),
               maximum_native_episodes=63, development_registry=str(ledger), coordinator=coordinator_identity,
               automatic_retries=0))
    completed = []
    try:
        for seed in SEEDS:
            worker_output = args.output/'scenes'/f'seed{seed}'
            argv = [sys.executable, str(Path(__file__).resolve()), '--design', str(args.design),
                    '--seed', str(seed), '--ledger', str(ledger), '--output', str(worker_output), '--execute']
            write_once(args.output/f'worker_intent_{seed}.json', dict(seed=seed, argv=argv,
                        coordinator=coordinator_identity, automatic_retries=0))
            update_state(args.output/'state.json', dict(status='running_scene', seed=seed, completed=completed))
            # The inherited outer resource guardian owns this process tree and
            # lease. Each scene is a fresh worker; no CUDA or solver state crosses.
            with (args.output/f'worker_{seed}.log').open('x') as log:
                result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode:
                raise RuntimeError(f'Scene worker{seed} failed({result.returncode}); no next scene/retry')
            scene = read(worker_output/'state.json')
            if scene['status'] not in {'scene_complete', 'blocked_reference', 'blocked_baseline'}:
                raise ValueError('Unverified scientific worker termination')
            from verifier import verify_scene
            verify_scene(worker_output, design, previous_claims=read(ledger)['claims'])
            completed.append(dict(seed=seed, status=scene['status']))
        from verifier import verify_campaign
        report = verify_campaign(args.output, design)
        write_once(args.output/'report.json', report)
        update_state(args.output/'state.json', dict(status='finite_development_complete', completed=completed,
                                                    formal_evidence_eligible=False, acceptance_ready=False))
        return 0
    except BaseException as exc:
        update_state(args.output/'state.json', dict(status='infrastructure_failed', error=type(exc).__name__+': '+str(exc),
                                                    completed=completed, automatic_retries=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    os.environ['ROBOTWIN_PATH'] = read(args.design)['native_root']
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
