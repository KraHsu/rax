#!/usr/bin/env python3
"""Finite actual-trajectory development discovery. Default mode is CPU check-only.

This entry point is intentionally separate from strict search/acceptance. A
lease-owning supervisor must launch it for native execution; it does not claim
or kill other GPU processes and never resumes an existing output directory.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.robotwin.acceptance import derived_values_equal, file_hash  # noqa: E402
from services.robotwin.acceptance_replay import (  # noqa: E402
    ALIGNMENT, SUCCESS_RULE, arrays, install_base_frame_adapter, quaternion_distance,
    replay, sample, tracking_report,
)
from services.robotwin.collection import atomic_json, initial_scene, scene_fingerprint  # noqa: E402
from services.robotwin.observed_discovery import (  # noqa: E402
    KIND, MANIFEST, PURPOSE, candidate_for, classify_observed,
    contained, runtime_fingerprint, validate_design, verify_observed_campaign,
)
from services.robotwin.success_metadata import MARKER, initialize_success_metadata  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text())


def artifact(root, path, value):
    target = contained(root, path)
    atomic_json(target, value)
    return dict(path=path, sha256=file_hash(target))


def execute_slot(root, manifest, slot, factory, reference, control):
    """One fresh native attempt; even an exception leaves immutable raw evidence."""
    root = Path(root)
    directory = root/'trials'/slot['trial_id']
    directory.mkdir(parents=True, exist_ok=False)
    candidate = candidate_for(root, slot, reference)
    raw = dict(trial_id=slot['trial_id'], scene_sha256=manifest['scene_sha256'],
               execution_complete=False, native_success=None, ever_success=False, final_success=False,
               planner_success=False, realized=False, alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE)
    actual = {arm: dict(pose=[], gripper=[]) for arm in ('left', 'right')}
    native, scenes, scene_errors = None, [], []
    started = time.monotonic()

    def observe(frame):
        try:
            scenes.append(dict(frame=frame, scene=initial_scene(native)))
        except Exception as exc:
            scene_errors.append(dict(frame=frame, error=f'{type(exc).__name__}: {exc}'))

    try:
        native = factory(manifest['scene_seed'])
        if scene_fingerprint(initial_scene(native)) != manifest['scene_sha256']:
            raise ValueError('Fresh scene differs from the frozen observed scene')
        if hasattr(native, MARKER):
            raw['success_metadata'] = getattr(native, MARKER)
        adapter = install_base_frame_adapter(native)
        if not derived_values_equal(adapter, reference['replay_adapter']):
            raise ValueError('Replay adapter differs from frozen source assets')
        measured = sample(native)
        expected = arrays(reference['trajectory'])
        for arm in ('left', 'right'):
            if (not np.allclose(measured[arm]['pose'][:3], expected[arm]['pose'][0, :3], rtol=0, atol=1e-5)
                    or quaternion_distance(measured[arm]['pose'][3:], expected[arm]['pose'][0, 3:]) > 3e-3
                    or abs(measured[arm]['gripper']-expected[arm]['gripper'][0]) > 1e-5):
                raise ValueError('Fresh measured robot state differs from reference')
        raw.update(replay_adapter=adapter, replay_adapter_applied=True)
        observe(0)
        original = native.take_action

        def observed_action(*args, **kwargs):
            value = original(*args, **kwargs)
            if int(native.take_action_cnt) % 4 == 0:
                observe(int(native.take_action_cnt))
            return value

        native.take_action = observed_action
        raw.update(replay(native, candidate['trajectory']))
        actual = raw.pop('actual')
        observe(len(actual['left']['pose'])-1)
        if raw['execution_complete']:
            tracking = tracking_report(actual, candidate['trajectory'], baseline=control,
                                       reference=reference['trajectory'] if control is not None else None)
            raw.update(tracking=tracking, realized=tracking['realized'])
    except Exception as exc:
        raw.update(error=f'{type(exc).__name__}: {exc}', execution_complete=False)
    finally:
        if native is not None:
            try:
                native.close_env()
            except Exception as exc:
                raw['cleanup_error'] = f'{type(exc).__name__}: {exc}'
    raw['seconds'] = time.monotonic()-started
    prefix = f'trials/{slot["trial_id"]}/'
    record = dict(trial_id=slot['trial_id'], manifest_sha256=file_hash(root/MANIFEST),
                  trajectory_sha256=slot['trajectory_sha256'],
                  trajectory=artifact(root, prefix+'trajectory.json',
                                      dict(commanded=candidate['trajectory'], actual=actual, alignment=ALIGNMENT)),
                  outcome=artifact(root, prefix+'outcome.json', raw),
                  scene_trace=artifact(root, prefix+'scene_trace.json',
                                       dict(purpose='diagnostic_only_not_constraint_or_training_input',
                                            stride=4, observations=scenes, errors=scene_errors)))
    receipt = artifact(root, prefix+'record.json', record)
    with (root/'observed_trials.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(receipt, trial_id=slot['trial_id']))+'\n')
        stream.flush()
        os.fsync(stream.fileno())
    return raw, actual, classify_observed(reference, candidate, raw, actual, control=control)


def execute_plan(root, manifest, reference, factory):
    """Failure stops this finite plan; no retries, seed scan, or adaptive selection."""
    root = Path(root)
    control = None
    for slot in manifest['trial_plan']:
        atomic_json(root/'state.json', dict(status='running', current_trial=slot['trial_id'], automatic_retries=0))
        raw, actual, result = execute_slot(root, manifest, slot, factory, reference, control)
        proceed = result['observationally_valid'] and not raw.get('error') and not raw.get('cleanup_error')
        if slot['kind'] == 'control':
            proceed = proceed and result['control_execution_reliable']
            control = actual
        if not proceed:
            atomic_json(root/'state.json', dict(status='blocked', trial_id=slot['trial_id'],
                                               reason='control_not_reliable_or_incomplete_execution', automatic_retries=0))
            return 2
    atomic_json(root/'state.json', dict(status='outcomes_complete', automatic_retries=0))
    return 0


def run(args):
    design_path = args.design.resolve()
    design = read(design_path)
    reference, _, exclusions = validate_design(design_path.parent, design)
    if not args.execute:
        print(json.dumps(dict(check='passed', native_slots=25, gpu_started=False, formal_evidence_eligible=False)))
        return 0
    from services.robotwin.runtime import initialize_runtime, native_config, source_commit
    from services.robotwin.collection import install_planner_cache
    from scripts.validate_robotwin_constraint_acceptance import training_exclusions
    if source_commit() != design['native_commit'] or training_exclusions(args.training_root) != exclusions:
        raise ValueError('Native source or training exclusion snapshot changed')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)  # Existing intent/output is never resumed, even after failure.
    for relative in design['source_files']:
        destination = contained(root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(contained(design_path.parent, relative), destination)
    shutil.copyfile(design_path, root/'design_plan.json')
    manifest = dict(version=1, kind=KIND, purpose=PURPOSE, task=design['task'],
                    scene_seed=design['seed'], scene_sha256=design['scene_sha256'], split='dev',
                    design_sha256=file_hash(root/'design_plan.json'), runtime_sha256=runtime_fingerprint(),
                    trial_plan=design['trial_plan'], formal_evidence_eligible=False,
                    acceptance_ready=False, publication_allowed=False, created_unix=time.time())
    atomic_json(root/MANIFEST, manifest)
    atomic_json(root/'state.json', dict(status='initializing', automatic_retries=0))
    try:
        initialize_runtime()
        install_planner_cache()
        cls = getattr(importlib.import_module('envs.'+design['task']), design['task'])
        kwargs = native_config(design['task'], 0)
        kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                      eval_video_save_dir=None, save_path=str(root/'native_scratch'))

        def factory(seed):
            random.seed(seed)
            native = cls()
            try:
                native.setup_demo(now_ep_num=0, seed=seed, is_test=False, **kwargs)
                initialize_success_metadata(design['task'], native, expected_scene_sha256=design['scene_sha256'])
                return native
            except BaseException:
                native.close_env()
                raise

        code = execute_plan(root, manifest, reference, factory)
        atomic_json(root/'observed_report.json', verify_observed_campaign(root))
        return code
    except BaseException as exc:
        atomic_json(root/'state.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', automatic_retries=0))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Native execution, only via a reviewed lease-owning supervisor')
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
