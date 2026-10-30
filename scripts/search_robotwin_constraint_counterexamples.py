#!/usr/bin/env python3
"""Search a frozen constraint set for executable native failures.

This is counterexample discovery, not acceptance or training publication. Expert
recordings supply scene identity and fixed label/activation anchors; proposals
may have entirely different motion and gripper behavior. All candidates and
native slots are sealed before simulation. Native outcomes never choose seeds.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from services.robotwin.acceptance import (  # noqa: E402
    _verify_instantiation,
    annotation_policy_fingerprint, derived_values_equal, evaluator_fingerprint,
    file_hash, json_hash, trajectory_violations, validate_acceptance_bundle,
    validate_replay_adapter,
)
from services.robotwin.acceptance_replay import (  # noqa: E402
    ALIGNMENT, BASE_FRAME_ADAPTER, SUCCESS_RULE, arrays, install_base_frame_adapter,
    quaternion_distance, replay, sample, tracking_report,
)
from services.robotwin.adversarial_search import (  # noqa: E402
    SEARCH_VERSION, generate_candidates, score_candidate, search_manifest, source_fingerprint,
)
from services.robotwin.collection import atomic_json, initial_scene, scene_fingerprint  # noqa: E402
from services.robotwin.programmatic_annotation import template_hash  # noqa: E402
from scripts.validate_robotwin_constraint_acceptance import (  # noqa: E402
    bind, load_template, reference_bundle, training_exclusions,
)

DISCOVERY_VERSION = 1
MODEL_KINDS = {'model_closed_loop', 'model_receding_horizon_snapshot'}
SCENE_TRACE_STRIDE = 4


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value)-set('0123456789abcdef')


def relative_artifact(root, relative):
    path = (Path(root)/relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError('Evidence path escapes the campaign or is missing')
    return path


def artifact(root, name, value):
    path = Path(root)/name
    atomic_json(path, value)
    return dict(path=str(path.relative_to(root)), sha256=file_hash(path))


def checked_artifact(root, spec):
    path = relative_artifact(root, spec['path'])
    if file_hash(path) != spec['sha256']:
        raise ValueError('Frozen artifact hash changed')
    return read(path)


def validate_split(plan, split, *, task, scene_seed, exclusions, template_sha256,
                   annotation_policy_sha256, release=None):
    if (plan.get('version') != 1 or plan.get('task') != task or split not in ('dev', 'holdout')):
        raise ValueError('Explicit task-specific dev/holdout split plan required')
    groups = {}
    for key in ('dev_seeds', 'holdout_seeds'):
        values = plan.get(key)
        if (not isinstance(values, list) or not values or any(type(v) is not int or v < 0 for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Both seed partitions must be nonempty and unique')
        groups[key] = set(values)
    if (groups['dev_seeds'] & groups['holdout_seeds']
            or (groups['dev_seeds'] | groups['holdout_seeds']) & set(exclusions['excluded_training_seeds'])):
        raise ValueError('Training, development and held-out seeds must be disjoint')
    if scene_seed not in groups['dev_seeds' if split == 'dev' else 'holdout_seeds']:
        raise ValueError('Reference seed is outside the preregistered partition')
    if split == 'holdout':
        expected = dict(version=1, task=task, template_sha256=template_sha256,
                        annotation_policy_sha256=annotation_policy_sha256,
                        split_plan_sha256=json_hash(plan))
        if not isinstance(release, dict) or any(release.get(k) != v for k, v in expected.items()):
            raise ValueError('Held-out rollout requires a matching frozen-template release')
        if (type(release.get('frozen_unix')) not in (int, float) or not math.isfinite(release['frozen_unix'])
                or not 0 < release['frozen_unix'] <= time.time()
                or not sha(release.get('development_report_sha256'))):
            raise ValueError('Held-out release lacks a prior freeze and development report identity')
    elif release is not None:
        raise ValueError('Development discovery must not consume a held-out release')
    return plan


def validate_holdout_configuration(config, generator_sha256, *, created_unix=None):
    """Do not select final native trials by their constraint-pass outcome."""
    if config['split'] != 'holdout':
        return
    generator = config['split_plan'].get('holdout_generator')
    release = config['holdout_release']
    keys = ('random_seed', 'candidate_budget', 'baseline_repeats', 'include_outside_controls')
    if (not isinstance(generator, dict) or generator.get('version') != 1
            or generator.get('generator_sha256') != generator_sha256
            or not isinstance(generator.get('configuration'), dict)
            or set(generator['configuration']) != set(keys)
            or json_hash(generator['configuration']) != json_hash({k: config[k] for k in keys})
            or config['include_outside_controls'] is not True):
        raise ValueError('Final holdout must freeze generator settings and replay all candidates including rejected controls')
    if (release.get('holdout_generator_sha256') != json_hash(generator)
            or config.get('feedback_inputs_sha256') is not None):
        raise ValueError('Held-out generator release changed or development feedback was supplied')
    if created_unix is not None and created_unix < release['frozen_unix']:
        raise ValueError('Held-out candidates predate the final template freeze')


def check_initial(reference, candidate, *, physical=False):
    ref, value = arrays(reference), arrays(candidate)
    atol = 1e-5 if physical else 1e-10
    for arm in ref:
        if ref[arm]['pose'].shape != value[arm]['pose'].shape:
            raise ValueError('Candidates must keep all reference indices; no padding, resampling or DTW')
        a, b = ref[arm], value[arm]
        if (not np.allclose(a['pose'][0, :3], b['pose'][0, :3], rtol=0, atol=atol)
                or quaternion_distance(a['pose'][0, 3:], b['pose'][0, 3:]) > (3e-3 if physical else 1e-9)
                or abs(a['gripper'][0]-b['gripper'][0]) > atol):
            raise ValueError('Candidate initial state differs from the frozen measured state')


def load_model_inputs(path, reference):
    if path is None:
        return []
    value = read(path)
    p = reference['provenance']
    expected = dict(version=1, task=p['task'], scene_seed=p['seed'], scene_sha256=p['scene_sha256'],
                    alignment=ALIGNMENT, action_contract=reference['bundle']['action_contract'])
    if any(value.get(k) != v for k, v in expected.items()) or not isinstance(value.get('candidates'), list):
        raise ValueError('Model inputs need the exact scene, contract and reference-command alignment')
    length = len(arrays(reference['trajectory'])['left']['pose'])
    ids = set()
    for row in value['candidates']:
        provenance = row.get('model_provenance')
        if (row.get('source') != 'model' or not isinstance(provenance, dict)
                or row.get('family') not in MODEL_KINDS or provenance.get('source_kind') != row['family']
                or not sha(provenance.get('checkpoint_sha256')) or not sha(provenance.get('source_run_sha256'))
                or not isinstance(row.get('candidate_id'), str) or not row['candidate_id']
                or row['candidate_id'] in ids):
            raise ValueError('Model provenance must identify real checkpoint/run/source kind and unique IDs')
        if row['family'] == 'model_receding_horizon_snapshot':
            embedding = provenance.get('embedding')
            if (provenance.get('full_policy_rollout') is not False
                    or not sha(provenance.get('raw_chunk_sha256'))
                    or not isinstance(embedding, dict)
                    or any(not isinstance(embedding.get(k), str) or not embedding[k]
                           for k in ('prefix_source', 'suffix_source'))
                    or type(embedding.get('chunk_start')) is not int or type(embedding.get('chunk_end')) is not int
                    or not 1 <= embedding['chunk_start'] < embedding['chunk_end'] <= length):
                raise ValueError('Snapshot imports must disclose their raw chunk and explicit full-N embedding')
        elif provenance.get('full_policy_rollout') is not True:
            raise ValueError('Closed-loop imports must identify a complete original fixed-N policy rollout')
        check_initial(reference['trajectory'], row['trajectory'])
        ids.add(row['candidate_id'])
    return value['candidates']


def prepare_reference(source, source_manifest, template, exclusions):
    """Preserve old source identity and instantiate an explicitly new revision.

    An old policy hash is never rewritten to pretend that the source was made
    by current code. Only this campaign's working labels use the current frozen
    template; the original reference and manifest are separately archived.
    """
    p = source['provenance']
    if (source_manifest.get('mode') != 'demo_clean' or source_manifest.get('split') != 'validation'
            or source_manifest.get('task') != template['task'] or p['task'] != template['task']
            or source_manifest.get('native_commit') != exclusions['native_commit']
            or source_manifest.get('alignment') != ALIGNMENT or source_manifest.get('native_success_rule') != SUCCESS_RULE
            or not sha(source_manifest.get('annotation_policy_sha256'))
            or source_manifest.get('template_sha256') != template_hash(source_manifest['template'])):
        raise ValueError('Source must be an identified clean native reference experiment')
    if (p['seed'] in exclusions['excluded_training_seeds']
            or p['scene_sha256'] in exclusions['excluded_training_scene_sha256']
            or p.get('mode') != 'demo_clean' or p.get('split') != 'validation'):
        raise ValueError('Reference overlaps training or is not held-out clean')
    if scene_fingerprint(source['initial_scene']) != p['scene_sha256']:
        raise ValueError('Source initial scene identity changed')
    if p.get('source_episode_sha256') != json_hash(dict(scene=source['initial_scene'], trajectory=source['trajectory'])):
        raise ValueError('Source trajectory/scene content changed')
    data = arrays(source['trajectory'])
    validate_acceptance_bundle(source['bundle'], len(data['left']['pose']))
    validate_replay_adapter(source['replay_adapter'])
    if (source.get('replay_adapter_applied') is not False
            or not derived_values_equal(source['replay_adapter'], source_manifest['replay_adapter'])):
        raise ValueError('Reference capture adapter identity changed')
    working = copy.deepcopy(source)
    working['bundle'], working['template_audit'] = reference_bundle(
        template, source['trajectory'], p['seed'], p['scene_sha256'])
    working['constraint_revision'] = dict(template_sha256=template_hash(template),
                                         annotation_policy_sha256=annotation_policy_fingerprint(),
                                         source_annotation_policy_sha256=source_manifest['annotation_policy_sha256'])
    baseline = trajectory_violations(working['bundle'], working['trajectory'], working['trajectory'])
    if not baseline['passed']:
        raise ValueError('Current frozen constraints reject their reference')
    return working


def load_feedback(path, reference):
    if path is None:
        return [], []
    value = read(path)
    if value.get('version') != 1 or not isinstance(value.get('campaigns'), list):
        raise ValueError('Feedback requires completed campaign roots and explicit trial IDs')
    parents, receipts = [], []
    for request in value['campaigns']:
        root = Path(request['campaign_root']).resolve()
        report = verify_campaign(root)
        if not report['complete'] or report['manifest']['split'] != 'dev':
            raise ValueError('Feedback may consume only complete development campaigns')
        old_ref = checked_artifact(root, report['manifest']['reference'])
        if (old_ref['provenance']['scene_sha256'] != reference['provenance']['scene_sha256']
                or old_ref['provenance']['seed'] != reference['provenance']['seed']
                or json_hash(old_ref['trajectory']) != json_hash(reference['trajectory'])):
            raise ValueError('Feedback parents must have the same exact scene and reference observations')
        ids = request['trial_ids']
        if not isinstance(ids, list) or len(set(ids)) != len(ids):
            raise ValueError('Feedback trial IDs must be explicit and unique')
        for trial_id in ids:
            row = next((r for r in report['trials'] if r['trial_id'] == trial_id), None)
            if row is None or not row['strong_counterexample']:
                raise ValueError('Feedback parent is not a verified inside-set native failure')
            record = read(root/'trials'/trial_id/'record.json')
            actual = checked_artifact(root, record['trajectory'])['actual']
            still_inside = trajectory_violations(reference['bundle'], reference['trajectory'], actual)['passed']
            receipts.append(dict(campaign_root=str(root), trial_id=trial_id, evidence_sha256=row['evidence_sha256'],
                                 manifest_sha256=report['manifest_sha256'], records_sha256=report['records_sha256'],
                                 still_inside_current_constraints=still_inside))
            if still_inside:
                parents.append(dict(evidence_id=row['evidence_sha256'], native_success=False, execution_complete=True,
                                    planner_success=True, realized=True, trajectory=actual))
    return parents, receipts


def make_trial_plan(candidates, baseline_repeats, *, include_outside_controls=False):
    rows = [dict(trial_id=f'baseline_{i}', kind='baseline', candidate_id=None, source='expert', family='baseline')
            for i in range(baseline_repeats)]
    for index, candidate in enumerate(candidates):
        if candidate['commanded_constraint_pass'] or include_outside_controls:
            rows.append(dict(trial_id=f'candidate_{index:04d}', kind='candidate',
                             candidate_id=candidate['candidate_id'], source=candidate['source'], family=candidate['family'],
                             candidate_index=index))
    if len(rows) == baseline_repeats:
        raise ValueError('No constraint-admissible candidates; discovery cannot spend only baseline rollouts')
    return rows


def _candidate(manifest, planned, root, reference):
    if planned['kind'] == 'baseline':
        return dict(trajectory=reference['trajectory'], source='expert', family='baseline', model_provenance=None)
    candidate = checked_artifact(root, manifest['candidates'][planned['candidate_index']])
    if (any(candidate[k] != planned[k] for k in ('candidate_id', 'source', 'family'))
            or json_hash(candidate['trajectory']) != candidate['trajectory_sha256']):
        raise ValueError('Planned candidate identity changed')
    check_initial(reference['trajectory'], candidate['trajectory'])
    fresh = score_candidate(reference['trajectory'], reference['bundle'], candidate['trajectory'])
    if fresh['commanded_constraint_pass'] is not candidate['commanded_constraint_pass']:
        raise ValueError('Candidate screening decision differs from its frozen constraints')
    return candidate


def classify_trial(reference, candidate, raw, actual, *, baseline=None):
    """Recompute three independent facts: label pass, execution, task outcome."""
    commanded = candidate['trajectory']
    check_initial(reference['trajectory'], commanded)
    command_score = trajectory_violations(reference['bundle'], reference['trajectory'], commanded)
    complete = raw.get('execution_complete') is True
    measured_score = None
    track = None
    planner_success = False
    reason = 'incomplete_execution'
    if complete:
        check_initial(reference['trajectory'], actual, physical=True)
        expected = {(arm, i) for arm in ('left', 'right') for i in range(1, len(arrays(commanded)['left']['pose']))}
        rows = raw['planner_results']
        planner_success = (len(rows) == len(expected) and {(r['arm'], r['frame']) for r in rows} == expected
                           and all(r['status'] == 'Success' for r in rows))
        if raw.get('planner_success') is not planner_success:
            raise ValueError('Planner summary differs from measured native records')
        if type(raw.get('native_success')) is not bool or raw.get('ever_success') is not raw['native_success']:
            raise ValueError('Native task result requires the independently recorded ever-success predicate')
        validate_replay_adapter(raw.get('replay_adapter'))
        if (raw.get('replay_adapter_applied') is not True
                or not derived_values_equal(raw['replay_adapter'], reference['replay_adapter'])):
            raise ValueError('Native replay adapter differs from the frozen scene/reference')
        track = tracking_report(actual, commanded, baseline=baseline,
                                reference=reference['trajectory'] if baseline is not None else None)
        if raw.get('realized') is not track['realized'] or not derived_values_equal(raw.get('tracking'), track):
            raise ValueError('Native execution/realization summary changed')
        measured_score = trajectory_violations(reference['bundle'], reference['trajectory'], actual)
        reason = None if planner_success and track['realized'] else 'planner_failure_or_unrealized_command'
    eligible = complete and planner_success and track is not None and track['realized']
    failure = eligible and raw['native_success'] is False
    counterexample = failure and command_score['passed']
    strong = counterexample and measured_score['passed']
    return dict(constraint_space='effective_native_command', raw_model_constraint_pass=None,
                execution_complete=complete, planner_success=planner_success,
                realized=bool(track is not None and track['realized']), eligible=bool(eligible), invalid_reason=reason,
                native_success=raw.get('native_success') if complete else None,
                commanded_constraint_pass=command_score['passed'],
                actual_constraint_pass=measured_score['passed'] if measured_score is not None else None,
                command_score=command_score, actual_score=measured_score, tracking=track,
                is_counterexample=bool(counterexample), strong_counterexample=bool(strong),
                command_only_counterexample=bool(counterexample and not strong),
                controller_failure=bool(command_score['passed'] and not eligible),
                successful_variation_rejected=bool(eligible and raw['native_success'] and not command_score['passed']),
                actual_success_rejected=bool(eligible and raw['native_success'] and not measured_score['passed']))


def rebuild_index(root):
    records = []
    for path in sorted((Path(root)/'trials').glob('*/record.json')):
        record = read(path)
        records.append(dict(trial_id=record['trial_id'], path=str(path.relative_to(root)), sha256=file_hash(path)))
    temporary = Path(root)/'trials.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in records))
    temporary.replace(Path(root)/'trials.jsonl')


def verify_trial(root, manifest, planned):
    root = Path(root)
    record_path = root/'trials'/planned['trial_id']/'record.json'
    record = read(record_path)
    reference = checked_artifact(root, manifest['reference'])
    candidate = _candidate(manifest, planned, root, reference)
    for key, value in dict(trial_id=planned['trial_id'], manifest_sha256=file_hash(root/'search_manifest.json'),
                           reference_sha256=manifest['reference']['sha256'],
                           trajectory_sha256=json_hash(candidate['trajectory'])).items():
        if record.get(key) != value:
            raise ValueError(f'Trial record context changed: {key}')
    raw = checked_artifact(root, record['outcome'])
    trace = checked_artifact(root, record['trajectory'])
    if 'scene_trace' in record:
        diagnostic = checked_artifact(root, record['scene_trace'])
        if diagnostic.get('purpose') != 'diagnostic_only_not_constraint_or_training_input':
            raise ValueError('Scene observations must remain separate from action-only constraint evaluation')
    if json_hash(trace['commanded']) != json_hash(candidate['trajectory']):
        raise ValueError('Executed command differs from the preregistered candidate')
    for key in ('trial_id', 'scene_sha256'):
        if raw.get(key) != (planned['trial_id'] if key == 'trial_id' else manifest['scene_sha256']):
            raise ValueError('Native outcome belongs to another trial or scene')
    baseline = None
    if planned['trial_id'] != 'baseline_0':
        first = next(p for p in manifest['trial_plan'] if p['trial_id'] == 'baseline_0')
        check = verify_trial(root, manifest, first)
        if not (check['eligible'] and check['native_success']):
            raise ValueError('Candidate has no reliable same-scene baseline')
        base_record = read(root/'trials/baseline_0/record.json')
        baseline = checked_artifact(root, base_record['trajectory'])['actual']
    result = classify_trial(reference, candidate, raw, trace['actual'], baseline=baseline)
    return dict(result, trial_id=planned['trial_id'], kind=planned['kind'], candidate_id=planned.get('candidate_id'),
                task=manifest['task'], split=manifest['split'], scene_seed=manifest['scene_seed'],
                scene_sha256=manifest['scene_sha256'], source=candidate['source'], family=candidate['family'],
                model_provenance=candidate.get('model_provenance'),
                trajectory_sha256=json_hash(candidate['trajectory']), actual_trajectory_sha256=json_hash(trace['actual']),
                reference_sha256=manifest['reference']['sha256'], bundle_sha256=json_hash(reference['bundle']),
                template_sha256=manifest['template_sha256'], annotation_policy_sha256=manifest['annotation_policy_sha256'],
                evidence_sha256=file_hash(record_path))


def verify_campaign(root):
    """Read and independently reassess every committed slot; never run a GPU."""
    root = Path(root)
    manifest = read(root/'search_manifest.json')
    if (manifest.get('version') != DISCOVERY_VERSION or manifest.get('purpose') != 'counterexample_discovery_not_acceptance'
            or manifest.get('evaluator_sha256') != evaluator_fingerprint()
            or manifest.get('alignment') != ALIGNMENT or manifest.get('native_success_rule') != SUCCESS_RULE):
        raise ValueError('Unsupported discovery evaluator/protocol identity')
    config = read(root/'configuration.json')
    if (json_hash(config) != manifest['configuration_sha256']
            or config['annotation_policy_sha256'] != manifest['annotation_policy_sha256']
            or config['template_sha256'] != manifest['template_sha256']):
        raise ValueError('Frozen campaign inputs changed')
    if (json_hash(manifest['split_plan']) != manifest['split_plan_sha256']
            or manifest['holdout_release_sha256'] != (json_hash(manifest['holdout_release']) if manifest['holdout_release'] is not None else None)):
        raise ValueError('Frozen split/release identity changed')
    if template_hash(manifest['template']) != manifest['template_sha256']:
        raise ValueError('Frozen template changed')
    reference = checked_artifact(root, manifest['reference'])
    validate_acceptance_bundle(reference['bundle'], len(arrays(reference['trajectory'])['left']['pose']))
    _verify_instantiation(reference, manifest['template'])
    if (manifest['scene_seed'] != reference['provenance']['seed']
            or manifest['scene_sha256'] != reference['provenance']['scene_sha256']
            or json_hash(reference['bundle']) != manifest['bundle_sha256']):
        raise ValueError('Frozen reference identity changed')
    validate_split(manifest['split_plan'], manifest['split'], task=manifest['task'], scene_seed=manifest['scene_seed'],
                   exclusions=manifest['training_exclusions'], template_sha256=manifest['template_sha256'],
                   annotation_policy_sha256=manifest['annotation_policy_sha256'], release=manifest['holdout_release'])
    validate_holdout_configuration(config, manifest['generator_sha256'], created_unix=manifest['created_unix'])
    # Sources are retained verbatim even when current data-only template policy differs.
    for key in ('source_reference', 'source_manifest'):
        checked_artifact(root, manifest[key])
    ids = [p['trial_id'] for p in manifest['trial_plan']]
    if len(ids) != len(set(ids)) or ids[:2] != ['baseline_0', 'baseline_1']:
        raise ValueError('Discovery needs preregistered unique slots and two leading baselines')
    candidates = [checked_artifact(root, spec) for spec in manifest['candidates']]
    expected_plan = make_trial_plan(candidates, config['baseline_repeats'],
                                    include_outside_controls=config['include_outside_controls'])
    if manifest['trial_plan'] != expected_plan:
        raise ValueError('Native trial slots differ from the complete frozen selection rule')
    for planned in manifest['trial_plan']:
        _candidate(manifest, planned, root, reference)
    index_path = root/'trials.jsonl'
    index = [json.loads(x) for x in index_path.read_text().splitlines() if x.strip()] if index_path.exists() else []
    received = [r['trial_id'] for r in index]
    if len(set(received)) != len(received) or set(received)-set(ids):
        raise ValueError('Unplanned or duplicate native outcomes cannot enter a campaign')
    rows = []
    for entry in index:
        path = relative_artifact(root, entry['path'])
        if file_hash(path) != entry['sha256'] or path != (root/'trials'/entry['trial_id']/'record.json').resolve():
            raise ValueError('Committed trial receipt changed')
        planned = next(p for p in manifest['trial_plan'] if p['trial_id'] == entry['trial_id'])
        rows.append(verify_trial(root, manifest, planned))
    variants = [r for r in rows if r['kind'] == 'candidate']
    complete = set(received) == set(ids)
    return dict(version=DISCOVERY_VERSION, purpose='counterexample_discovery_not_acceptance',
                manifest_sha256=file_hash(root/'search_manifest.json'), records_sha256=file_hash(index_path) if index_path.exists() else None,
                manifest=manifest, trials=rows, complete=complete,
                counts=dict(planned=len(ids), received=len(rows), candidates=len(variants),
                            valid_candidates=sum(r['eligible'] for r in variants),
                            command_inside_candidates=sum(r['commanded_constraint_pass'] for r in variants),
                            counterexamples=sum(r['is_counterexample'] for r in variants),
                            strong_counterexamples=sum(r['strong_counterexample'] for r in variants),
                            controller_failures=sum(r['controller_failure'] for r in variants),
                            baseline_false_rejects=sum(r['actual_success_rejected'] for r in rows if r['kind']=='baseline')),
                acceptance_ready=False,
                meaning='Adversarially selected finite search outcomes; not an IID success estimate or a publication certificate.')


def execute_trial(root, manifest, planned, factory):
    reference = checked_artifact(root, manifest['reference'])
    candidate = _candidate(manifest, planned, root, reference)
    directory = root/'trials'/planned['trial_id']
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    raw = dict(native_success=None, planner_success=False, realized=False, execution_complete=False,
               ever_success=False, final_success=False, alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE)
    actual = {'left': {'pose': [], 'gripper': []}, 'right': {'pose': [], 'gripper': []}}
    scene_trace, scene_errors = [], []
    native = None
    def observe_scene(frame):
        try:
            value = dict(frame=frame, scene=initial_scene(native))
            if scene_trace and scene_trace[-1]['frame'] == frame:
                scene_trace[-1] = value
            else:
                scene_trace.append(value)
        except Exception as exc:
            # Diagnostics must not turn a successful controller call into an
            # infrastructure failure. The missing observation remains explicit.
            scene_errors.append(dict(frame=frame, error=f'{type(exc).__name__}: {exc}'))
    try:
        native = factory(manifest['scene_seed'])
        if scene_fingerprint(initial_scene(native)) != manifest['scene_sha256']:
            raise ValueError('Fresh scene differs from the frozen reference')
        adapter = install_base_frame_adapter(native)
        if not derived_values_equal(adapter, manifest['replay_adapter']):
            raise ValueError('Native replay assets differ from the frozen adapter')
        measured = sample(native)
        for arm in ('left', 'right'):
            expected = arrays(reference['trajectory'])[arm]
            if (not np.allclose(measured[arm]['pose'][:3], expected['pose'][0,:3], rtol=0, atol=1e-5)
                    or quaternion_distance(measured[arm]['pose'][3:], expected['pose'][0,3:]) > 3e-3
                    or abs(measured[arm]['gripper']-expected['gripper'][0]) > 1e-5):
                raise ValueError('Fresh robot initial state differs from reference')
        raw.update(replay_adapter=adapter, replay_adapter_applied=True)
        observe_scene(0)
        if hasattr(native, 'take_action'):
            original_take_action = native.take_action
            def observed_take_action(*args, **kwargs):
                result = original_take_action(*args, **kwargs)
                frame = int(native.take_action_cnt)
                if frame % SCENE_TRACE_STRIDE == 0:
                    observe_scene(frame)
                return result
            native.take_action = observed_take_action
        raw.update(replay(native, candidate['trajectory']))
        actual = raw.pop('actual')
        observe_scene(len(actual['left']['pose'])-1)
        if raw['execution_complete']:
            baseline = None
            if planned['trial_id'] != 'baseline_0':
                base_record = read(root/'trials/baseline_0/record.json')
                baseline = checked_artifact(root, base_record['trajectory'])['actual']
            track = tracking_report(actual, candidate['trajectory'], baseline=baseline,
                                    reference=reference['trajectory'] if baseline is not None else None)
            raw.update(realized=track['realized'], tracking=track)
    except Exception as exc:
        raw.update(error=f'{type(exc).__name__}: {exc}', execution_complete=False)
    finally:
        if native is not None:
            native.close_env()
    raw.update(trial_id=planned['trial_id'], scene_sha256=manifest['scene_sha256'], seconds=time.monotonic()-started)
    trace = artifact(root, str(directory.relative_to(root)/'trajectory.json'),
                     dict(actual=actual, commanded=candidate['trajectory'], alignment=ALIGNMENT))
    outcome = artifact(root, str(directory.relative_to(root)/'outcome.json'), raw)
    scenes = artifact(root, str(directory.relative_to(root)/'scene_trace.json'),
                      dict(purpose='diagnostic_only_not_constraint_or_training_input',
                           stride=SCENE_TRACE_STRIDE, observations=scene_trace, errors=scene_errors))
    record = dict(trial_id=planned['trial_id'], manifest_sha256=file_hash(root/'search_manifest.json'),
                  reference_sha256=manifest['reference']['sha256'], trajectory_sha256=json_hash(candidate['trajectory']),
                  trajectory=trace, outcome=outcome, scene_trace=scenes)
    atomic_json(directory/'record.json', record)
    rebuild_index(root)


def prepare_campaign(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    source_path = args.reference.resolve()
    source_manifest_path = (args.reference_manifest or source_path.parent.parent/'experiment_manifest.json').resolve()
    source, source_manifest = read(source_path), read(source_manifest_path)
    template = load_template(args.template)
    exclusions = training_exclusions(args.training_root)
    matches = [r for r in source_manifest['reference_artifacts'] if r['seed'] == source['provenance']['seed']]
    if len(matches) != 1 or matches[0]['sha256'] != file_hash(source_path):
        raise ValueError('Reference is not the exact sealed source artifact')
    working = prepare_reference(source, source_manifest, template, exclusions)
    split_plan = read(args.split_plan)
    release = read(args.holdout_release) if args.holdout_release else None
    if args.split == 'holdout':
        from services.robotwin.refinement_exposure import require_active_reservation
        require_active_reservation(release)
    validate_split(split_plan, args.split, task=template['task'], scene_seed=working['provenance']['seed'], exclusions=exclusions,
                   template_sha256=template_hash(template), annotation_policy_sha256=annotation_policy_fingerprint(), release=release)
    if args.split == 'holdout' and args.feedback:
        raise ValueError('Held-out evaluation cannot consume development outcome feedback')
    models = load_model_inputs(args.model_candidates, working)
    parents, feedback = load_feedback(args.feedback, working)
    config = dict(version=DISCOVERY_VERSION, task=template['task'], template_sha256=template_hash(template),
                  annotation_policy_sha256=annotation_policy_fingerprint(), source_reference_sha256=file_hash(source_path),
                  source_manifest_sha256=file_hash(source_manifest_path), split=args.split, split_plan=split_plan,
                  holdout_release=release, random_seed=args.random_seed, candidate_budget=args.candidate_budget,
                  baseline_repeats=args.baseline_repeats, include_outside_controls=args.include_outside_controls,
                  scene_trace_stride=SCENE_TRACE_STRIDE,
                  search_version=SEARCH_VERSION, search_source_sha256=source_fingerprint(),
                  replay_adapter_version=BASE_FRAME_ADAPTER,
                  driver_sha256=file_hash(__file__), evaluator_sha256=evaluator_fingerprint(),
                  model_inputs_sha256=file_hash(args.model_candidates) if args.model_candidates else None,
                  feedback_inputs_sha256=file_hash(args.feedback) if args.feedback else None, feedback_receipts=feedback,
                  training_exclusions_sha256=json_hash(exclusions))
    validate_holdout_configuration(config, source_fingerprint())
    bind(root/'configuration.json', config)
    if (root/'search_manifest.json').exists():
        manifest = read(root/'search_manifest.json')
        if manifest['configuration_sha256'] != json_hash(config):
            raise ValueError('Frozen campaign configuration changed')
        verify_campaign(root)
        return manifest
    candidates = generate_candidates(working['trajectory'], working['bundle'], seed=args.random_seed,
                                     budget=args.candidate_budget, model_candidates=models, feedback_parents=parents)
    candidate_specs = [artifact(root, f'candidates/candidate_{i:04d}.json', row) for i, row in enumerate(candidates)]
    manifest = dict(version=DISCOVERY_VERSION, purpose='counterexample_discovery_not_acceptance',
                    created_unix=time.time(), generator_sha256=source_fingerprint(),
                    configuration_sha256=json_hash(config), task=template['task'], template=template,
                    template_sha256=template_hash(template), annotation_policy_sha256=annotation_policy_fingerprint(),
                    source_annotation_policy_sha256=source_manifest['annotation_policy_sha256'],
                    native_commit=exclusions['native_commit'], alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE,
                    evaluator_sha256=evaluator_fingerprint(), search_version=SEARCH_VERSION,
                    split=args.split, split_plan=split_plan, split_plan_sha256=json_hash(split_plan), holdout_release=release,
                    feedback_used=args.feedback is not None,
                    holdout_release_sha256=json_hash(release) if release is not None else None,
                    holdout_generator_sha256=json_hash(split_plan['holdout_generator']) if 'holdout_generator' in split_plan else None,
                    scene_seed=working['provenance']['seed'], scene_sha256=working['provenance']['scene_sha256'],
                    training_exclusions=exclusions, replay_adapter=working['replay_adapter'],
                    reference=artifact(root, 'reference.json', working), bundle_sha256=json_hash(working['bundle']),
                    source_reference=artifact(root, 'source_reference.json', source),
                    source_manifest=artifact(root, 'source_manifest.json', source_manifest),
                    candidates=candidate_specs, candidate_search=search_manifest(working['trajectory'], working['bundle'],
                    seed=args.random_seed, budget=args.candidate_budget, candidates=candidates,
                    model_candidates=models, feedback_parents=parents),
                    trial_plan=make_trial_plan(candidates, args.baseline_repeats, include_outside_controls=args.include_outside_controls))
    # Preserve original source bytes, including their original hashes, not a
    # re-serialized object masquerading as the old source artifact.
    (root/'source_reference.json').write_bytes(source_path.read_bytes())
    (root/'source_manifest.json').write_bytes(source_manifest_path.read_bytes())
    manifest['source_reference']['sha256'] = file_hash(root/'source_reference.json')
    manifest['source_manifest']['sha256'] = file_hash(root/'source_manifest.json')
    bind(root/'search_manifest.json', manifest)
    rebuild_index(root)
    return manifest


def run(args):
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    with (root/'.search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare_campaign(args)
        report = verify_campaign(root)
        if args.check_only or report['complete']:
            atomic_json(root/'counterexample_summary.json', report)
            print(json.dumps(dict(check='passed', gpu_started=False, complete=report['complete'],
                                  planned_trials=report['counts']['planned'], purpose=report['purpose'])))
            return 0
        if (root/'run.json').exists() and read(root/'run.json').get('status') in ('failed', 'blocked'):
            raise ValueError('Failed discovery run is immutable; inspect evidence and use a new run directory')
        from services.robotwin.runtime import initialize_runtime, native_config, source_commit
        from services.robotwin.collection import install_planner_cache
        if source_commit() != manifest['native_commit']:
            raise ValueError('Native source revision differs from the frozen reference')
        initialize_runtime(); install_planner_cache()
        native_class = getattr(importlib.import_module('envs.'+manifest['task']), manifest['task'])
        kwargs = native_config(manifest['task'], 0)
        kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                      eval_video_save_dir=None, save_path=str(root/'native_scratch'))
        def factory(seed):
            random.seed(seed)
            native = native_class()
            try:
                native.setup_demo(now_ep_num=0, seed=seed, is_test=False, **kwargs)
            except BaseException:
                native.close_env(); raise
            return native
        started = time.time()
        if (root/'run.json').exists():
            started = read(root/'run.json').get('started_unix', started)
        state = dict(status='running_trials', started_unix=started, task=manifest['task'], split=manifest['split'])
        try:
            for planned in manifest['trial_plan']:
                if manifest['split'] == 'holdout':
                    from services.robotwin.refinement_exposure import require_active_reservation
                    require_active_reservation(manifest['holdout_release'])
                directory = root/'trials'/planned['trial_id']
                state.update(current_trial=planned['trial_id'], updated_unix=time.time()); atomic_json(root/'run.json', state)
                if not (directory/'record.json').exists():
                    if directory.exists():
                        raise RuntimeError('Incomplete native trial requires review; never silently resampled')
                    execute_trial(root, manifest, planned, factory)
                checked = verify_trial(root, manifest, planned)
                if planned['kind'] == 'baseline' and not (checked['eligible'] and checked['native_success']):
                    state.update(status='blocked', reason='baseline_execution_unreliable')
                    atomic_json(root/'run.json', state)
                    atomic_json(root/'counterexample_summary.json', verify_campaign(root))
                    return 2
                state.update(completed_trials=len(list((root/'trials').glob('*/record.json'))),
                             last_trial=planned['trial_id'], updated_unix=time.time())
                atomic_json(root/'run.json', state)
                atomic_json(root/'counterexample_summary.json', verify_campaign(root))
                print(json.dumps(dict(state, counterexample=checked['is_counterexample'],
                                      strong_counterexample=checked['strong_counterexample'])), flush=True)
            state.update(status='outcomes_complete', finished_unix=time.time()); atomic_json(root/'run.json', state)
            return 0
        except BaseException as exc:
            state.update(status='failed', error=f'{type(exc).__name__}: {exc}', updated_unix=time.time())
            atomic_json(root/'run.json', state)
            raise


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'template', 'training-root', 'output', 'split-plan'):
        value.add_argument('--'+name, type=Path, required=True)
    for name in ('reference-manifest', 'holdout-release', 'model-candidates', 'feedback'):
        value.add_argument('--'+name, type=Path)
    value.add_argument('--split', choices=['dev','holdout'], default='dev')
    value.add_argument('--random-seed', type=int, default=20261030)
    value.add_argument('--candidate-budget', type=int, default=24)
    value.add_argument('--baseline-repeats', type=int, default=2)
    value.add_argument('--include-outside-controls', action='store_true')
    value.add_argument('--check-only', action='store_true')
    return value


def main():
    value = parser(); args = value.parse_args()
    if (args.random_seed < 0 or not 1 <= args.candidate_budget <= 128 or not 2 <= args.baseline_repeats <= 5):
        value.error('Invalid frozen random seed or native budget')
    raise SystemExit(run(args))


if __name__ == '__main__':
    main()
