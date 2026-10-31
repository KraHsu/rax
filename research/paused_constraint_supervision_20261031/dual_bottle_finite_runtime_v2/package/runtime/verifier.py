"""Recompute finite prefix gates, all action scores and original predicate evidence."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from candidate_design import fixed_reference, generate
from contracts import SEEDS, digest, file_hash, new_development_ledger, read
from predicate_observer import verify_observations
from driver import (PACKAGE, checked_artifact, classify_trial, reference_bundle,
                    frozen_manifest, samples_to_trajectory, scene_fingerprint, selected_candidates)
from services.robotwin.acceptance import trajectory_violations


def baseline_ok(result):
    return bool(result['eligible'] and result['native_success'] and result['commanded_constraint_pass']
                and result['actual_constraint_pass'])


def reporting_flags(candidate, result):
    baseline = candidate.get('category', 'baseline') == 'baseline'
    outside_control = bool(candidate.get('declared_outside_control', False))
    if not baseline:
        inside = bool(result['commanded_constraint_pass'])
        if candidate['commanded_domain'] != ('inside' if inside else 'outside'):
            raise ValueError('Command domain was not frozen from identical scored command')
        if outside_control != (candidate['category'] == 'outside_set_control'):
            raise ValueError('Declared outside control identity changed')
    else:
        inside = bool(result['commanded_constraint_pass'])
    admissible_evidence = not baseline and inside and not outside_control
    return dict(baseline=baseline, declared_outside_control=outside_control,
                commanded_domain='inside' if inside else 'outside',
                reportable_strong_counterexample=bool(admissible_evidence and result['strong_counterexample']),
                reportable_command_only_counterexample=bool(admissible_evidence and result['command_only_counterexample']),
                scope='Predeclared outside controls and baselines never count as crafted inside-set counterexamples.')


def capture_evidence_hashes(root):
    names = ['capture_intent.json', 'capture_attempt.json', 'capture_observations.json.gz',
             'preflight.json', 'state.json']
    return {name: file_hash(root/name) if (root/name).exists() else None for name in names}


def verify_scene(root, design, *, previous_claims):
    root = Path(root)
    state, attempt, intent = read(root/'state.json'), read(root/'capture_attempt.json'), read(root/'capture_intent.json')
    seed = state['seed']
    if (seed not in SEEDS or attempt['seed'] != seed or intent['seed'] != seed
            or intent['native_initializations_started'] != 1 or intent['expert_attempts_started'] != 0
            or attempt['expert_attempts_started'] not in (0, 1) or attempt['automatic_retries'] != 0
            or 'cleanup_error' in attempt or state['dataset_version_sha256'] != design['dataset_version_sha256']):
        raise ValueError('Capture budget/context/cleanup differs')
    preflight = read(root/'preflight.json')
    if preflight['dataset_version_sha256'] != design['dataset_version_sha256']:
        raise ValueError('Dataset provenance changed')
    scene_sha = attempt.get('scene_sha256')
    if scene_sha is not None and scene_fingerprint(attempt['initial_scene']) != scene_sha:
        raise ValueError('Captured scene hash changed')
    if attempt.get('development_ledger_after_claim_sha256') is not None:
        indices = [i for i, row in enumerate(previous_claims) if row['seed'] == seed and row['scene_sha256'] == scene_sha]
        if len(indices) != 1 or scene_sha in preflight['excluded_scene_sha256']:
            raise ValueError('Fresh claim is missing/duplicated or excluded')
        prefix = new_development_ledger(design['identity_sha256'])
        prefix['claims'] = previous_claims[:indices[0]+1]
        if digest(prefix) != attempt['development_ledger_after_claim_sha256']:
            raise ValueError('Pre-expert scene-claim ledger prefix changed')
    elif attempt['expert_attempts_started'] != 0:
        raise ValueError('Expert actions preceded a committed scene claim')
    observations = json.loads(gzip.decompress((root/'capture_observations.json.gz').read_bytes())) if (root/'capture_observations.json.gz').exists() else None
    if observations is not None:
        verify_observations(observations)
        final_rows = [r for r in observations['predicate_rle'] if r['data']['phase'] == 'original_capture_final_check']
        final_calls = sum(r['last_call']-r['first_call']+1 for r in final_rows)
        if observations['observation_complete']:
            if attempt['planner_success'] is True and attempt.get('rejected_episode') is None:
                if final_calls != 1 or final_rows[0]['data']['native_result'] is not attempt['native_success']:
                    raise ValueError('Unique original capture final predicate result differs')
            elif final_calls or attempt['native_success'] is not None:
                raise ValueError('Planner failure must preserve skipped final predicate asNone')
    trial_dirs = sorted((root/'trials').iterdir()) if (root/'trials').exists() else []
    if any(not p.is_dir() or not (p/'record.json').is_file() for p in trial_dirs):
        raise ValueError('Extra/partial trial directory cannot be a scientific completion')
    if state['status'] == 'blocked_reference':
        if (attempt['status'] != 'blocked_reference' or trial_dirs or (root/'reference.json').exists()
                or (root/'manifest.json').exists() or (root/'generation.json.gz').exists()
                or state['reason'] != attempt['stop_reason']):
            raise ValueError('Blocked reference contains execution or inconsistent reason')
        reason = attempt['stop_reason']
        if reason == 'reference_native_expert_failed':
            if attempt['expert_attempts_started'] != 1 or not (attempt['planner_success'] is False or attempt['native_success'] is False):
                raise ValueError('No original native failure supporting reference stop')
        elif reason.startswith('Unsupported'):
            if attempt['planner_success'] is not True or attempt['native_success'] is not True:
                raise ValueError('Unsupported signature needs an actually successful reference')
            try:
                fixed_reference(samples_to_trajectory(attempt['samples']))
            except ValueError as exc:
                if str(exc) != reason:
                    raise ValueError('Reference signature stop reason changed') from exc
            else:
                raise ValueError('Reference signature actually supported')
        elif reason.startswith('Fresh scene duplicates'):
            previous = {r['scene_sha256'] for r in previous_claims if r['seed'] != seed}
            if attempt['expert_attempts_started'] != 0 or scene_sha not in (set(preflight['excluded_scene_sha256']) | previous):
                raise ValueError('Scene duplicate stop lacks an actual collision')
        elif reason.startswith('native_expert_rejected: '):
            if attempt.get('rejected_episode') not in {
                    'native_plan_failed_before_grasp',
                    "native_grasp_pose_unavailable: 'NoneType' object is not subscriptable",
                    'native_grasp_pose_unavailable: target_pose cannot be None for move action.'}:
                raise ValueError('Unknown expert rejection cannot become a scientific stop')
            if attempt['expert_attempts_started'] != 1:
                raise ValueError('No expert attempt supporting rejection')
        else:
            raise ValueError('Unknown scientific stop classification')
        return dict(seed=seed, status=state['status'], native_initializations=1, expert_attempts=attempt['expert_attempts_started'],
                    trials=[], reason=reason, evidence_sha256=capture_evidence_hashes(root),
                    observation_complete=observations is not None and observations['observation_complete'])
    if attempt['status'] != 'captured' or attempt['planner_success'] is not True or attempt['native_success'] is not True:
        raise ValueError('Replay phase lacks a valid unique capture')
    own_claims = [r for r in previous_claims if r['seed'] == seed]
    if len(own_claims) != 1 or own_claims[0]['scene_sha256'] != scene_sha:
        raise ValueError('Fresh scene is not uniquely bound in development ledger')
    reference = read(root/'reference.json')
    if (samples_to_trajectory(attempt['samples']) != reference['trajectory']
            or reference['initial_scene'] != attempt['initial_scene']
            or reference['provenance']['seed'] != seed
            or reference['provenance']['scene_sha256'] != scene_sha
            or reference['provenance']['source_episode_sha256'] != digest(dict(scene=reference['initial_scene'], trajectory=reference['trajectory']))):
        raise ValueError('Reference not equal to captured samples/scene')
    fixed_reference(reference['trajectory'])
    bundle, audit = reference_bundle(read(PACKAGE/'design/v3/template.json'), reference['trajectory'], seed, scene_sha)
    if bundle != reference['bundle'] or digest(audit) != digest(reference['template_audit']):
        raise ValueError('Reference labels do not compile from frozen template')
    generation = json.loads(gzip.decompress((root/'generation.json.gz').read_bytes()))
    if digest(generate(reference['trajectory'], seed, scene_sha)) != digest(generation):
        raise ValueError('Candidate raw slots/scores/commands differ from fixed CPU generator')
    expected = selected_candidates(generation)
    manifest = read(root/'manifest.json')
    if digest(manifest) != digest(frozen_manifest(root, design, seed, generation)):
        raise ValueError('Frozen manifest identities/budget changed')
    plans = [('baseline_0', dict(source='expert', family='baseline', trajectory=reference['trajectory'])),
             ('baseline_1', dict(source='expert', family='baseline', trajectory=reference['trajectory']))]
    plans += [(f"candidate_{c['slot']:02d}", c) for c in expected]
    actual_ids = {p.name for p in trial_dirs}
    if state['status'] == 'scene_complete':
        expected_ids = {p[0] for p in plans}
    elif state['status'] == 'blocked_baseline' and state['blocked_trial'] in {'baseline_0', 'baseline_1'}:
        expected_ids = {'baseline_0'} if state['blocked_trial'] == 'baseline_0' else {'baseline_0', 'baseline_1'}
    else:
        raise ValueError('Scene is neither complete nor a valid scientific stop')
    if actual_ids != expected_ids:
        raise ValueError('Trial directory set differs from exact permitted prefix')
    comparators = {'selected_V3': reference['bundle']}
    for name, filename in [('A20', 'template.json'), ('A15', 'comparator_A15.json'),
                           ('original_template_current_compiler', 'original_active_template.json')]:
        comparators[name] = reference_bundle(read(PACKAGE/'design/original'/filename), reference['trajectory'], seed, scene_sha)[0]
    rows, baseline = [], None
    for trial_id, candidate in plans:
        if trial_id not in actual_ids:
            continue
        trial = root/'trials'/trial_id
        record, intent = read(trial/'record.json'), read(trial/'intent.json')
        if (record['trial_id'] != trial_id or record['seed'] != seed
                or record['reference_sha256'] != digest(reference)
                or record['candidate_sha256'] != digest(candidate)
                or record['infrastructure_failure'] is not False
                or intent['commanded_sha256'] != digest(candidate['trajectory'])
                or intent['native_initializations_started'] != 1 or intent['automatic_retries'] != 0):
            raise ValueError('Trial identity/initialization/infrastructure differs')
        trace = checked_artifact(root, record['trajectory'])
        raw = checked_artifact(root, record['outcome'])
        if trace['commanded'] != candidate['trajectory'] or raw['trial_id'] != trial_id or raw['scene_sha256'] != scene_sha:
            raise ValueError('Frozen command or native context changed')
        result = classify_trial(reference, candidate, raw, trace['actual'], baseline=baseline)
        if digest(result) != digest(record['classification']):
            raise ValueError('Stored scores/realization differ from exact recomputation')
        observed = checked_artifact(root, record['observations']) if record['observations'] else None
        if observed is None:
            raise ValueError('Completed native episode has no observation artifact')
        checked = verify_observations(observed)
        if observed['observation_complete']:
            if raw['execution_complete'] and checked['any_recorded_success'] is not raw['native_success']:
                raise ValueError('Predicate timeline differs from native outcome')
            if raw['execution_complete'] and (observed['last_return'] is None
                    or observed['last_return']['data']['native_result'] is not raw['final_success']):
                raise ValueError('Original final predicate differs from final_success')
        scores = {}
        for name, comparator in comparators.items():
            scores[name] = {'commanded': trajectory_violations(comparator, reference['trajectory'], trace['commanded']),
                            'actual': trajectory_violations(comparator, reference['trajectory'], trace['actual']) if raw['execution_complete'] else None}
        reporting = reporting_flags(candidate, result)
        rows.append(dict(trial_id=trial_id, category=candidate.get('category', 'baseline'),
                         reporting=reporting,
                         family=candidate['family'], source=candidate['source'], result=result,
                         native_final_success=raw.get('final_success'),
                         comparator_scores=scores, observation_complete=observed['observation_complete'],
                         record_sha256=file_hash(trial/'record.json')))
        if trial_id.startswith('baseline_'):
            valid = baseline_ok(result)
            if state['status'] == 'blocked_baseline' and trial_id == state['blocked_trial']:
                if valid:
                    raise ValueError('Claimed blocked baseline actually passes')
            elif not valid:
                raise ValueError('Later episode followed an invalid baseline')
            if trial_id == 'baseline_0':
                baseline = trace['actual']
    return dict(seed=seed, status=state['status'], scene_sha256=scene_sha,
                native_initializations=1+len(rows), expert_attempts=1, trials=rows,
                evidence_sha256=capture_evidence_hashes(root),
                observation_complete=observations is not None and observations['observation_complete'] and all(r['observation_complete'] for r in rows),
                manifest_sha256=file_hash(root/'manifest.json'), unavailable_slots=manifest['unavailable_slots'])


def verify_campaign(root, design):
    root = Path(root)
    intent = read(root/'intent.json')
    ledger_path = root/'development_registry'/'pick_dual_bottles.json'
    ledger = read(ledger_path)
    if (intent['design_sha256'] != design['identity_sha256'] or intent['seeds'] != list(SEEDS)
            or ledger['seeds'] != list(SEEDS) or ledger['plan_sha256'] != design['identity_sha256']
            or ledger['kind'] != 'local_development_exposure_not_final_holdout_registry'):
        raise ValueError('Coordinator/development ledger identities changed')
    scenes = list((root/'scenes').iterdir())
    if {p.name for p in scenes} != {f'seed{s}' for s in SEEDS}:
        raise ValueError('All3 fixed scene jobs must terminate without replacement')
    claims = ledger['claims']
    if len({r['seed'] for r in claims}) != len(claims) or len({r['scene_sha256'] for r in claims}) != len(claims):
        raise ValueError('Repeated seed or scene in current development campaign')
    if [r['seed'] for r in claims] != sorted(r['seed'] for r in claims):
        raise ValueError('Scene claims do not follow the fixed worker order')
    worker_ids = []
    for seed in SEEDS:
        start = read(root/f'worker_intent_{seed}.json')
        receipt = read(root/'worker_receipts'/f'seed{seed}.json')
        if (start['coordinator'] != intent['coordinator'] or receipt['coordinator'] != intent['coordinator']
                or start['seed'] != seed or receipt['seed'] != seed
                or receipt['design_sha256'] != design['identity_sha256']
                or start['automatic_retries'] != 0 or receipt['automatic_retries'] != 0):
            raise ValueError('Worker process/parent/seed identity changed')
        worker_ids.append(digest(receipt['worker']))
    if len(set(worker_ids)) != 3:
        raise ValueError('Scene workers are not three distinct fresh process identities')
    rows = [verify_scene(root/'scenes'/f'seed{s}', design, previous_claims=claims) for s in SEEDS]
    trials = [r for scene in rows for r in scene['trials']]
    candidates = [r for r in trials if r['category'] != 'baseline']
    baselines = [r for r in trials if r['category'] == 'baseline']
    count = sum(s['native_initializations'] for s in rows)
    if count > 63 or any(s['native_initializations'] > 21 for s in rows):
        raise ValueError('Finite episode cap exceeded')
    counts = dict(native_initializations=count, expert_attempts=sum(s['expert_attempts'] for s in rows),
        baseline_replays=len(baselines), crafted_replays=len(candidates),
        strict_strong_failures=sum(r['reporting']['reportable_strong_counterexample'] for r in candidates),
        strict_command_only_failures=sum(r['reporting']['reportable_command_only_counterexample'] for r in candidates),
        command_admitted_controller_failures=sum(r['result']['controller_failure'] for r in candidates),
        command_admitted_controller_native_successes=sum(r['result']['controller_failure'] and r['result']['native_success'] is True for r in candidates),
        command_admitted_controller_native_failures=sum(r['result']['controller_failure'] and r['result']['native_success'] is False for r in candidates),
        command_admitted_controller_incomplete=sum(r['result']['controller_failure'] and r['result']['native_success'] is None for r in candidates),
        observed_actual_pass_native_failures=sum(r['result']['native_success'] is False and r['result']['actual_constraint_pass'] is True for r in candidates),
        observed_actual_pass_eligible_native_failures=sum(r['result']['eligible'] and r['result']['native_success'] is False and r['result']['actual_constraint_pass'] is True for r in candidates),
        observed_actual_pass_tracking_invalid_failures=sum(not r['result']['eligible'] and r['result']['native_success'] is False and r['result']['actual_constraint_pass'] is True for r in candidates),
        all_completed_native_successes=sum(r['result']['native_success'] is True for r in candidates),
        all_success_actual_rejections=sum(r['result']['native_success'] is True and r['result']['actual_constraint_pass'] is False for r in candidates),
        tracking_invalid_native_successes=sum(not r['result']['eligible'] and r['result']['native_success'] is True for r in candidates))
    baseline_counts = dict(replays=len(baselines), valid=sum(baseline_ok(r['result']) for r in baselines),
        native_successes=sum(r['result']['native_success'] is True for r in baselines),
        native_failures=sum(r['result']['native_success'] is False for r in baselines),
        tracking_or_planner_ineligible=sum(not r['result']['eligible'] for r in baselines),
        commanded_rejections=sum(r['result']['commanded_constraint_pass'] is False for r in baselines),
        actual_rejections=sum(r['result']['actual_constraint_pass'] is False for r in baselines),
        native_failures_with_both_constraints_pass=sum(r['result']['native_success'] is False
            and r['result']['commanded_constraint_pass'] and r['result']['actual_constraint_pass'] for r in baselines))
    baseline_diagnostics = [dict(seed=scene['seed'], trial_id=r['trial_id'], result=r['result'],
        record_sha256=r['record_sha256'], native_final_success=r['native_final_success'])
        for scene in rows for r in scene['trials'] if r['category'] == 'baseline']
    domain_strata = {}
    for name, selected in [('inside_challenges', [r for r in candidates if r['reporting']['commanded_domain'] == 'inside' and not r['reporting']['declared_outside_control']]),
                           ('outside_challenges', [r for r in candidates if r['reporting']['commanded_domain'] == 'outside' and not r['reporting']['declared_outside_control']]),
                           ('declared_outside_controls', [r for r in candidates if r['reporting']['declared_outside_control']])]:
        domain_strata[name] = dict(replays=len(selected), native_successes=sum(r['result']['native_success'] is True for r in selected),
            native_failures=sum(r['result']['native_success'] is False for r in selected),
            tracking_or_planner_invalid=sum(not r['result']['eligible'] for r in selected),
            actual_success_rejections=sum(r['result']['native_success'] is True and r['result']['actual_constraint_pass'] is False for r in selected),
            strict_strong=sum(r['reporting']['reportable_strong_counterexample'] for r in selected),
            observed_actual_pass_native_failures=sum(r['result']['native_success'] is False and r['result']['actual_constraint_pass'] is True for r in selected),
            observed_actual_pass_eligible_native_failures=sum(r['result']['eligible'] and r['result']['native_success'] is False and r['result']['actual_constraint_pass'] is True for r in selected),
            observed_actual_pass_invalid_native_failures=sum(not r['result']['eligible'] and r['result']['native_success'] is False and r['result']['actual_constraint_pass'] is True for r in selected),
            native_ever_true_final_false=sum(r['result']['native_success'] is True and r['native_final_success'] is False for r in selected))
    return dict(version=2, status='finite_development_verified', domain_strata=domain_strata, scene_reports=rows, counts=counts,
                counts_scope='Quality/success/failure counts use crafted candidates only. Capture and baseline episode budgets are separate; baseline diagnostics never count as crafted counterexamples.',
                baseline_counts=baseline_counts, baseline_diagnostics=baseline_diagnostics,
                dataset_version_sha256=design['dataset_version_sha256'], development_ledger_sha256=file_hash(ledger_path),
                observation_complete=all(s['observation_complete'] for s in rows),
                formal_evidence_eligible=False, acceptance_ready=False, publication_allowed=False, training_allowed=False,
                limitations='3 new development scenes, reference-conditioned crafted trajectories, no model rollout, no final holdout. Selected V3 still has3known strong failures across exposed old/new scenes. Every18rawslot is retained including commanded outside; scientific stops reduce coverage without replacement. No alpha projection or native feedback selection.')
