#!/usr/bin/env python3
"""Collect held-out clean native outcomes for versioned constraint acceptance.

Run only after the current benchmark releases its GPUs. This writes tiny JSON
reference/command/actual traces, not training images or randomized recordings.
Native expert success selects reference scenes; policy outcomes never select
or replace seeds. Baseline replay failure blocks the task instead of becoming
a negative example. Existing run configuration and planned trials are immutable.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.robotwin.acceptance_replay import (  # noqa: E402
    ALIGNMENT, SUCCESS_RULE, TRACKING, arrays, default_cases, intervene,
    replay, sample, samples_to_trajectory, tracking_report,
)
from services.robotwin.collection import atomic_json, initial_scene, scene_fingerprint  # noqa: E402
from services.robotwin.constraint_schema import HDF5_KEY, SCHEMA_VERSION, canonical_json  # noqa: E402
from services.robotwin.programmatic_annotation import instantiate, template_hash, validate_template  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def bind(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Frozen acceptance configuration changed: {path}')
    else:
        atomic_json(path, value)


def load_template(path):
    value = json.loads(Path(path).read_text())
    template = value.get('template', value)
    validate_template(template)
    return template


def training_exclusions(root):
    """Read only small provenance fields, never copy the training recordings."""
    import h5py
    seeds, scenes, commits = set(), set(), set()
    paths = sorted({*Path(root).glob('*/aloha-agilex_clean_50/data/episode*.hdf5'),
                    *Path(root).glob('*/aloha-agilex_clean_50/raw/episode*.hdf5')})
    if not paths:
        raise ValueError('No clean training recordings found for held-out exclusion')
    for path in paths:
        with h5py.File(path, 'r') as handle:
            meta = json.loads(handle.attrs['robotwin_expert_provenance'])
        if meta.get('collection') != 0 or meta.get('task_config') != 'demo_clean':
            raise ValueError('Exclusion source contains non-clean recordings')
        seeds.add(int(meta['seed'])); scenes.add(meta['initial_scene_sha256']); commits.add(meta['source_commit'])
    if len(commits) != 1:
        raise ValueError('Mixed native commits in the training source')
    return dict(excluded_training_seeds=sorted(seeds), excluded_training_scene_sha256=sorted(scenes),
                native_commit=commits.pop(), source_episodes=len(paths))


class TraceHandle(dict):
    """In-memory structural adapter for the existing numeric template engine."""


def reference_bundle(template, trajectory, seed, fingerprint):
    # Existing template math accepts only the training schema. That schema is
    # used for this in-memory structural adapter, then discarded: the artifact
    # is explicitly validation, and normal read_embedded rejects it as training.
    data = arrays(trajectory)
    bundle = dict(schema_version=SCHEMA_VERSION,
                  action_contract=dict(position='absolute', rotation='absolute', quaternion_order='wxyz',
                                       gripper='absolute_open_fraction', frame='world'),
                  provenance=dict(split='train', variant='clean_50', native_success=True,
                                  recording_id=f'acceptance-{template["task"]}-{seed}-{fingerprint[:12]}',
                                  source='acceptance_reference_not_for_training'), constraints=[])
    handle = TraceHandle({HDF5_KEY: np.asarray(canonical_json(bundle))})
    handle.attrs = {'robotwin_expert_provenance': json.dumps(dict(task=template['task'], seed=seed))}
    for arm, values in data.items():
        handle[f'endpose/{arm}_endpose'] = values['pose']
        handle[f'endpose/{arm}_gripper'] = values['gripper']
    constraints, audit = instantiate(handle, template)
    bundle['constraints'] = constraints
    bundle['provenance']['split'] = 'validation'
    bundle['provenance']['variant'] = 'clean_heldout'
    return bundle, audit


def capture_reference(factory, template, seed, excluded_scenes):
    from scripts.collect_robotwin_clean50 import play_expert
    native = factory(seed)
    started = time.monotonic()
    try:
        scene = initial_scene(native)
        fingerprint = scene_fingerprint(scene)
        if fingerprint in excluded_scenes:
            raise ValueError('Held-out scene duplicates a training scene')
        samples = [sample(native)]
        native._take_picture = lambda *args, **kwargs: samples.append(sample(native))
        play_expert(native)
        samples.append(sample(native))
        if not native.plan_success or not native.check_success():
            raise ValueError('Reference native expert did not succeed')
        trajectory = samples_to_trajectory(samples)
        bundle, audit = reference_bundle(template, trajectory, seed, fingerprint)
        source_hash = json_digest(dict(scene=scene, trajectory=trajectory))
        return dict(bundle=bundle, trajectory=trajectory, template_audit=audit, initial_scene=scene,
                    provenance=dict(task=template['task'], seed=seed, scene_sha256=fingerprint,
                                    mode='demo_clean', split='validation', source_episode_sha256=source_hash,
                                    recording_method='native_expert_save_freq_trace_no_images'),
                    capture_seconds=time.monotonic()-started)
    finally:
        native.close_env()


def planned_trials(references, baseline_repeats, cases):
    return [dict(trial_id=f'seed{seed}_{case["id"]}', seed=seed, variant_family=case['variant_family'], case=case)
            for seed in references
            for case in [*[dict(id=f'baseline_{i}', variant_family='baseline') for i in range(baseline_repeats)], *cases]]


def verify_plan(manifest, config, exclusions, template, output):
    """A resumed plan must still represent the exact preregistered experiment."""
    for key in ('task', 'template_sha256', 'annotation_policy_sha256', 'native_commit', 'alignment', 'native_success_rule'):
        if manifest[key] != config[key]:
            raise ValueError(f'Manifest/configuration mismatch: {key}')
    if manifest['template'] != template:
        raise ValueError('Manifest template changed')
    for key in ('excluded_training_seeds', 'excluded_training_scene_sha256'):
        if manifest[key] != exclusions[key]:
            raise ValueError('Training exclusion inventory changed')
    references = manifest['reference_artifacts']
    seeds = [entry['seed'] for entry in references]
    if len(seeds) != config['seeds'] or len(set(seeds)) != len(seeds):
        raise ValueError('Reference selection differs from the declared seed count')
    if manifest['trial_plan'] != planned_trials(seeds, config['baseline_repeats'], config['cases']):
        raise ValueError('Trial plan differs from the fixed intervention design')
    for entry in references:
        path = (output/entry['path']).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != entry['sha256']:
            raise ValueError('Frozen reference artifact changed')


def rebuild_index(output):
    """Recover an interrupted index write using only fully committed records."""
    records = [json.loads(p.read_text()) for p in sorted((output/'trials').glob('*/record.json'))]
    temporary = output/'trials.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
    temporary.replace(output/'trials.jsonl')


def trial_record(output, manifest, planned, reference_path, factory, baseline):
    reference = json.loads(reference_path.read_text())
    commanded = intervene(reference['trajectory'], planned['case'])
    directory = output/'trials'/planned['trial_id']
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    native = None
    raw = dict(native_success=None, planner_success=False, realized=False, execution_complete=False,
               ever_success=False, final_success=False, alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE)
    actual = {'left': {'pose': [], 'gripper': []}, 'right': {'pose': [], 'gripper': []}}
    try:
        native = factory(planned['seed'])
        fingerprint = scene_fingerprint(initial_scene(native))
        if fingerprint != reference['provenance']['scene_sha256']:
            raise ValueError('Fresh replay initial scene differs from the captured reference')
        measured = sample(native)
        for arm in ('left', 'right'):
            expected = np.asarray(reference['trajectory'][arm]['pose'][0])
            actual_initial = np.asarray(measured[arm]['pose'])
            if (not np.allclose(actual_initial[:3], expected[:3], rtol=0, atol=1e-5)
                    or abs(float(actual_initial[3:] @ expected[3:])) < 1-1e-6
                    or abs(measured[arm]['gripper']-reference['trajectory'][arm]['gripper'][0]) > 1e-5):
                raise ValueError('Fresh replay initial EEF/gripper state differs from reference')
        raw.update(replay(native, commanded))
        actual = raw.pop('actual')
        if raw['execution_complete']:
            track = tracking_report(actual, commanded, baseline=baseline,
                                    reference=reference['trajectory'] if baseline is not None else None)
            raw.update(realized=track['realized'], tracking=track)
        raw['initial_scene_match'] = True
    except Exception as exc:
        raw.update(error=f'{type(exc).__name__}: {exc}', execution_complete=False)
    finally:
        if native is not None:
            native.close_env()
    raw['seconds'] = time.monotonic()-started
    raw['trial_id'] = planned['trial_id']
    raw['case'] = planned['case']
    raw.update(task=manifest['task'], seed=planned['seed'],
               scene_sha256=reference['provenance']['scene_sha256'], variant_family=planned['variant_family'])
    atomic_json(directory/'trajectory.json', dict(actual=actual, commanded=commanded, alignment=ALIGNMENT))
    atomic_json(directory/'outcome.json', raw)
    record = {key: manifest[key] for key in ('version', 'task', 'template_sha256', 'annotation_policy_sha256',
               'native_commit', 'mode', 'split', 'alignment', 'native_success_rule')}
    record.update(trial_id=planned['trial_id'], seed=planned['seed'], variant_family=planned['variant_family'],
                  scene_sha256=reference['provenance']['scene_sha256'],
                  ref_episode_sha256=reference['provenance']['source_episode_sha256'],
                  bundle_sha256=json_digest(reference['bundle']))
    for name, path in [('reference', reference_path), ('trajectory', directory/'trajectory.json'),
                       ('outcome', directory/'outcome.json')]:
        record[f'{name}_path'] = str(path.relative_to(output))
        record[f'{name}_sha256'] = digest(path)
    atomic_json(directory/'record.json', record)
    # Committed record paths, never a half-written JSONL row, are the source of
    # truth for resume. Rebuild the small index atomically after every trial.
    rebuild_index(output)
    return record, raw, actual


def run(args):
    from services.robotwin.acceptance import annotation_policy_fingerprint, assess_evidence, evaluate_trial, validate_manifest
    from services.robotwin.runtime import initialize_runtime, native_config, source_commit
    from services.robotwin.collection import install_planner_cache

    template = load_template(args.template)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    exclusions = training_exclusions(args.training_root)
    native_revision = source_commit()
    if native_revision != exclusions['native_commit']:
        raise ValueError('Acceptance/native training source commit differs')
    config = dict(task=template['task'], template_sha256=template_hash(template),
                  annotation_policy_sha256=annotation_policy_fingerprint(), native_commit=native_revision,
                  seed_start=args.seed_start, seeds=args.seeds, max_seed_attempts=args.max_seed_attempts,
                  baseline_repeats=args.baseline_repeats, cases=default_cases(), tracking=TRACKING,
                  driver_sha256=digest(__file__), replay_sha256=digest(ROOT/'services/robotwin/acceptance_replay.py'),
                  exclusions_sha256=json_digest(exclusions), alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE)
    bind(output/'configuration.json', config)
    if args.check_only:
        print(json.dumps(dict(check='passed', gpu_started=False, trials_per_seed=args.baseline_repeats+len(default_cases()))))
        return 0
    lock = (output/'.runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run_state = dict(status='initializing', started_unix=time.time(), task=template['task'])
    if (output/'run.json').exists():
        previous = json.loads((output/'run.json').read_text())
        run_state['started_unix'] = previous['started_unix']
        if previous['status'] == 'blocked':
            raise ValueError('Blocked acceptance is immutable; fix its adapter/template in a new run')
        if previous['status'] == 'outcomes_complete':
            manifest = json.loads((output/'experiment_manifest.json').read_text())
            verify_plan(manifest, config, exclusions, template, output)
            report = assess_evidence(output)
            atomic_json(output/'acceptance_report.json', report)
            if report['errors'] or report['counts']['received'] != report['counts']['planned']:
                raise ValueError('Completed acceptance artifacts no longer validate')
            print(json.dumps(dict(status='outcomes_complete', gpu_started=False, ready=report['ready'])))
            return 0
    atomic_json(output/'run.json', run_state)
    initialize_runtime()
    install_planner_cache()
    native_class = getattr(importlib.import_module('envs.'+template['task']), template['task'])
    kwargs = native_config(template['task'], 0)
    kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                  eval_video_save_dir=None, save_path=str(output/'native_scratch'))

    def factory(seed):
        random.seed(seed)
        native = native_class()
        try:
            native.setup_demo(now_ep_num=0, seed=seed, is_test=False, **kwargs)
        except BaseException:
            native.close_env()
            raise
        return native

    manifest_path = output/'experiment_manifest.json'
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            references = {}
            excluded = set(exclusions['excluded_training_seeds'])
            scenes = set(exclusions['excluded_training_scene_sha256'])
            run_state['status'] = 'capturing_references'; atomic_json(output/'run.json', run_state)
            for seed in range(args.seed_start, args.seed_start+args.max_seed_attempts):
                if seed in excluded:
                    continue
                path = output/'references'/f'seed{seed}.json'
                try:
                    if path.exists():
                        reference = json.loads(path.read_text())
                    else:
                        reference = capture_reference(factory, template, seed, scenes)
                        atomic_json(path, reference)
                    fingerprint = reference['provenance']['scene_sha256']
                    if fingerprint in scenes:
                        raise ValueError('Reference scene duplicates a training/selected scene')
                    references[seed] = path
                    scenes.add(fingerprint)
                except Exception as exc:
                    atomic_json(output/'reference_rejections'/f'seed{seed}.json',
                                dict(seed=seed, reason=f'{type(exc).__name__}: {exc}'))
                if len(references) == args.seeds:
                    break
            if len(references) != args.seeds:
                raise RuntimeError('Insufficient distinct expert-success held-out references in the fixed seed scan')
            manifest = {key: config[key] for key in ('task', 'template_sha256', 'annotation_policy_sha256',
                        'native_commit', 'alignment', 'native_success_rule')}
            manifest.update(version=1, mode='demo_clean', split='validation', template=template,
                            excluded_training_seeds=exclusions['excluded_training_seeds'],
                            excluded_training_scene_sha256=exclusions['excluded_training_scene_sha256'],
                            trial_plan=planned_trials(references, args.baseline_repeats, config['cases']),
                            reference_artifacts=[dict(seed=seed, path=str(path.relative_to(output)), sha256=digest(path))
                                                 for seed, path in references.items()])
            bind(manifest_path, manifest)
        validate_manifest(manifest)
        verify_plan(manifest, config, exclusions, template, output)
        rebuild_index(output)
        baselines = {}
        run_state['status'] = 'running_trials'; atomic_json(output/'run.json', run_state)
        for planned in manifest['trial_plan']:
            directory = output/'trials'/planned['trial_id']
            record_path = directory/'record.json'
            if record_path.exists():
                record = json.loads(record_path.read_text())
                raw = json.loads((directory/'outcome.json').read_text())
                actual = json.loads((directory/'trajectory.json').read_text())['actual']
            else:
                if directory.exists():
                    raise RuntimeError(f'Incomplete trial needs review, never silently resampled: {directory}')
                record, raw, actual = trial_record(output, manifest, planned,
                                               output/'references'/f'seed{planned["seed"]}.json',
                                               factory, baselines.get(planned['seed']))
            checked = evaluate_trial(record, output, manifest)
            if planned['variant_family'] == 'baseline':
                if not (raw.get('native_success') and raw.get('planner_success') and raw.get('realized')
                        and raw.get('execution_complete') and checked.get('constraint_pass')):
                    run_state.update(status='blocked', reason='baseline_replay_not_valid', trial_id=planned['trial_id'])
                    atomic_json(output/'run.json', run_state)
                    return 2
                baselines.setdefault(planned['seed'], actual)
            run_state.update(completed_trials=len(list((output/'trials').glob('*/record.json'))),
                             last_trial=planned['trial_id'], last_seconds=raw['seconds'], updated_unix=time.time())
            atomic_json(output/'run.json', run_state)
            print(json.dumps(run_state), flush=True)
        run_state.update(status='outcomes_complete', finished_unix=time.time())
        atomic_json(output/'run.json', run_state)
        atomic_json(output/'acceptance_report.json', assess_evidence(output))
        return 0
    except BaseException as exc:
        run_state.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_unix=time.time())
        atomic_json(output/'run.json', run_state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--training-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed-start', default=10000, type=int)
    parser.add_argument('--seeds', default=3, type=int)
    parser.add_argument('--baseline-repeats', default=2, type=int)
    parser.add_argument('--max-seed-attempts', default=30, type=int)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.seed_start < 0 or not 1 <= args.seeds <= 10 or args.baseline_repeats < 2 or args.max_seed_attempts < args.seeds:
        parser.error('Invalid seed/reference/baseline limits')
    args.template = args.template.resolve(); args.training_root = args.training_root.resolve()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
