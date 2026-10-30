"""Data-only counterexample-guided constraint revisions and publication gate.

Discovery evidence never activates a template. A frozen revision is assessed on
new scene seeds with a frozen candidate generator. Risk bounds count scenes,
not correlated/adaptively selected trajectories. No simulator is imported here.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

from services.robotwin.acceptance import annotation_policy_fingerprint, derived_values_equal, file_hash, json_hash
from services.robotwin.collection import atomic_json
from services.robotwin.programmatic_annotation import template_hash, validate_template

VERSION = 1
DEFAULT_PROTOCOL = dict(version=1, confidence=.95, maximum_scene_failure_risk=.1,
                        minimum_accepted_scenes=29, minimum_native_success_scenes=10,
                        minimum_success_retention=.8, required_sources=['crafted', 'model'],
                        minimum_accepted_execution_fraction=.8,
                        minimum_crafted_families=3, maximum_rules=24, maximum_rule_growth=8,
                        require_zero_observed_false_accepts=True)
ACTIVATION_FIELDS = ('selector', 'subjects', 'start_anchor', 'required')
SOURCES = ('expert', 'crafted', 'model')


def read(path):
    return json.loads(Path(path).read_text())


def sha(value):
    return isinstance(value, str) and len(value) == 64 and not (set(value)-set('0123456789abcdef'))


def validate_protocol(protocol=None):
    p = copy.deepcopy(DEFAULT_PROTOCOL if protocol is None else protocol)
    if set(p) != set(DEFAULT_PROTOCOL) or p['version'] != VERSION:
        raise ValueError('Explicit complete refinement protocol required')
    for name in ('confidence', 'maximum_scene_failure_risk', 'minimum_success_retention', 'minimum_accepted_execution_fraction'):
        if type(p[name]) not in (int, float) or not math.isfinite(p[name]) or not 0 < p[name] < 1:
            raise ValueError('Invalid refinement probability')
    for name in ('minimum_accepted_scenes', 'minimum_native_success_scenes', 'minimum_crafted_families',
                 'maximum_rules', 'maximum_rule_growth'):
        if type(p[name]) is not int or p[name] < 1:
            raise ValueError('Invalid refinement count')
    if (p['maximum_rules'] > 24 or p['required_sources'] != ['crafted', 'model']
            or type(p['require_zero_observed_false_accepts']) is not bool):
        raise ValueError('Refinement requires crafted and actual model sources within the rule DSL')
    return p


def validate_split_plan(plan, *, training_seeds=()):
    if (plan.get('version') != VERSION or not isinstance(plan.get('task'), str)
            or not plan['task']):
        raise ValueError('Versioned task split plan required')
    for name in ('dev_seeds', 'holdout_seeds'):
        rows = plan.get(name)
        if (not isinstance(rows, list) or not rows or len(set(rows)) != len(rows)
                or any(type(x) is not int or x < 0 for x in rows)):
            raise ValueError('Explicit distinct dev and holdout scene seeds required')
    dev, holdout, train = map(set, (plan['dev_seeds'], plan['holdout_seeds'], training_seeds))
    if dev & holdout or train & (dev | holdout):
        raise ValueError('Training, development and final holdout seeds overlap')
    return plan


def validate_generator(value):
    if (not isinstance(value, dict) or set(value) != {'version', 'generator_sha256', 'configuration', 'sources'}
            or value['version'] != VERSION or not sha(value['generator_sha256'])
            or not isinstance(value['configuration'], dict) or set(value['sources']) != {'crafted', 'model'}):
        raise ValueError('Final holdout requires a frozen generator configuration and source budgets')
    json_hash(value)  # Reject non-JSON/nonfinite configuration without executing it.
    config = value['configuration']
    if (set(config) != {'random_seed', 'candidate_budget', 'baseline_repeats', 'include_outside_controls'}
            or type(config['random_seed']) is not int or config['random_seed'] < 0
            or type(config['candidate_budget']) is not int or not 1 <= config['candidate_budget'] <= 128
            or type(config['baseline_repeats']) is not int or not 2 <= config['baseline_repeats'] <= 5
            or config['include_outside_controls'] is not True):
        raise ValueError('Freeze generator parameters and replay rejected controls for success-retention assessment')
    for source, spec in value['sources'].items():
        expected = {'candidates_per_scene'} if source == 'crafted' else {
            'candidates_per_scene', 'checkpoint_sha256', 'source_kind'}
        if (set(spec) != expected or type(spec['candidates_per_scene']) is not int
                or spec['candidates_per_scene'] < 1):
            raise ValueError('Explicit fixed candidate budget per scene/source required')
        if source == 'model' and (not sha(spec['checkpoint_sha256'])
                or spec['source_kind'] != 'model_closed_loop'):
            raise ValueError('Final model holdout requires a frozen checkpoint and full model_closed_loop source')
    return value


def validate_ledger_row(row):
    required = ('trial_id', 'task', 'split', 'scene_seed', 'scene_sha256', 'source', 'family',
                'native_success', 'execution_complete', 'planner_success', 'realized',
                'commanded_constraint_pass', 'actual_constraint_pass', 'trajectory_sha256',
                'reference_sha256', 'template_sha256', 'annotation_policy_sha256', 'evidence_sha256')
    if any(k not in row for k in required) or row['source'] not in SOURCES or row['split'] not in ('dev', 'holdout'):
        raise ValueError('Incomplete verified trajectory ledger row')
    for name in ('trial_id', 'task', 'scene_sha256', 'family'):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError('Missing trajectory identity')
    if type(row['scene_seed']) is not int or row['scene_seed'] < 0:
        raise ValueError('Invalid scene seed')
    for name in ('execution_complete', 'planner_success', 'realized', 'commanded_constraint_pass'):
        if type(row[name]) is not bool:
            raise ValueError('Explicit execution and commanded-pass booleans required')
    eligible = row['execution_complete'] and row['planner_success'] and row['realized']
    for name in ('native_success', 'actual_constraint_pass'):
        if type(row[name]) is not bool and not (not eligible and row[name] is None):
            raise ValueError('Eligible native trials require independent outcome and actual-pass booleans')
    if any(not sha(row[k]) for k in required if k.endswith('_sha256') and k != 'scene_sha256'):
        raise ValueError('Frozen artifact and implementation SHA256 identities required')
    if row['source'] == 'model':
        m = row.get('model_provenance', {})
        if (not sha(m.get('checkpoint_sha256')) or not sha(m.get('source_run_sha256'))
                or m.get('source_kind') not in ('model_closed_loop', 'model_receding_horizon_snapshot')):
            raise ValueError('Model source requires actual checkpoint/run identities and collection mode')
        expected_full = m['source_kind'] == 'model_closed_loop'
        if m.get('full_policy_rollout') is not expected_full:
            raise ValueError('Model provenance must distinguish full policy rollouts from chunk proxies')
    return row


def eligible(row):
    return row['execution_complete'] and row['planner_success'] and row['realized']


def classify(row):
    validate_ledger_row(row)
    valid = eligible(row)
    false_accept = bool(valid and row['commanded_constraint_pass'] and not row['native_success'])
    return dict(eligible=valid, false_accept=false_accept,
                strong_false_accept=bool(false_accept and row['actual_constraint_pass']),
                false_reject=bool(valid and row['native_success'] and not row['commanded_constraint_pass']))


def source_summary(rows):
    counts = dict(received=len(rows), eligible=0, accepted=0, native_successes=0,
                  false_accepts=0, strong_false_accepts=0, false_rejects=0,
                  observed_actual_false_accepts=0)
    scenes = {}
    for row in rows:
        decision = classify(row)
        # Descriptive measured evidence survives failed intervention tracking.
        # This does not enter the precise-command causal/acceptance denominator.
        counts['observed_actual_false_accepts'] += bool(
            row['execution_complete'] and row['actual_constraint_pass'] is True and row['native_success'] is False)
        if not decision['eligible']:
            continue
        counts['eligible'] += 1
        counts['accepted'] += row['commanded_constraint_pass']
        counts['native_successes'] += row['native_success']
        for key, label in (('false_accepts', 'false_accept'), ('strong_false_accepts', 'strong_false_accept'),
                           ('false_rejects', 'false_reject')):
            counts[key] += decision[label]
        scene = scenes.setdefault((row['scene_seed'], row['scene_sha256']),
                                  dict(accepted=False, false_accept=False, native_success=False, accepted_success=False))
        scene['accepted'] |= row['commanded_constraint_pass']
        scene['false_accept'] |= decision['false_accept']
        scene['native_success'] |= row['native_success']
        scene['accepted_success'] |= row['native_success'] and row['commanded_constraint_pass']
    counts.update(accepted_scenes=sum(s['accepted'] for s in scenes.values()),
                  failed_accepted_scenes=sum(s['false_accept'] for s in scenes.values()),
                  native_success_scenes=sum(s['native_success'] for s in scenes.values()),
                  accepted_success_scenes=sum(s['accepted_success'] for s in scenes.values()))
    counts['success_retention'] = ((counts['native_successes']-counts['false_rejects'])/counts['native_successes']
                                   if counts['native_successes'] else None)
    counts['source_model_modes'] = sorted({r.get('model_provenance', {}).get('source_kind')
                                          for r in rows if r['source'] == 'model'})
    groups = {}
    for row in rows:
        groups.setdefault((row['scene_seed'], row['scene_sha256']), {}).setdefault(row['trajectory_sha256'], []).append(row)
    normalized, aliases, unique = [], [], 0
    for (seed, scene), trajectories in groups.items():
        successes = retained = accepted = executed = executed_all = 0
        for digest, repeats in trajectories.items():
            unique += 1
            if len({r['commanded_constraint_pass'] for r in repeats}) != 1:
                raise ValueError('Identical scene/action has inconsistent deterministic constraint verdicts')
            valid = [r for r in repeats if eligible(r)]
            native = {r['native_success'] for r in valid}
            accepted += repeats[0]['commanded_constraint_pass']
            fully_executed = all(eligible(r) for r in repeats)
            executed += repeats[0]['commanded_constraint_pass'] and fully_executed
            executed_all += fully_executed
            successes += True in native
            retained += True in native and repeats[0]['commanded_constraint_pass']
            if len(native) > 1:
                aliases.append(dict(scene_seed=seed, scene_sha256=scene, trajectory_sha256=digest,
                                    trials=[r['trial_id'] for r in repeats],
                                    meaning='Identical actions have different native outcomes; action-only rules cannot separate them.'))
        normalized.append(dict(scene_seed=seed, scene_sha256=scene, unique_trajectories=len(trajectories),
                               success_retention=retained/successes if successes else None,
                               accepted_execution_fraction=executed/accepted if accepted else None,
                               execution_fraction=executed_all/len(trajectories)))
    retention = [r['success_retention'] for r in normalized if r['success_retention'] is not None]
    execution = [r['accepted_execution_fraction'] for r in normalized if r['accepted_execution_fraction'] is not None]
    counts.update(unique_scene_trajectories=unique, duplicate_trajectory_rows=len(rows)-unique,
                  action_outcome_aliases=aliases, per_scene=normalized,
                  scene_normalized_success_retention=sum(retention)/len(retention) if retention else None,
                  minimum_scene_accepted_execution_fraction=min(execution) if execution else None)
    return counts


def development_summary(rows):
    """Summarize already verified discovery rows without any confidence claim."""
    if not rows or any(validate_ledger_row(r)['split'] != 'dev' for r in rows):
        raise ValueError('Development summary accepts only verified development rows')
    if len({(r['task'], r['trial_id']) for r in rows}) != len(rows):
        raise ValueError('Duplicate development trial IDs')
    if len({r['task'] for r in rows}) != 1:
        raise ValueError('Refinement campaign is task-specific')
    return dict(version=VERSION, task=rows[0]['task'], split='dev', confidence_eligible=False,
                records_sha256=json_hash(rows), rows=rows,
                sources={source: source_summary([r for r in rows if r['source'] == source]) for source in SOURCES},
                counterexamples=[r['trial_id'] for r in rows if classify(r)['false_accept']],
                meaning='Adaptive discovery counts only; no IID confidence or activation claim.')


def campaign_rows(verified):
    """Namespace scene-local runner IDs before aggregating verified campaigns."""
    digest = verified['manifest_sha256']
    if not sha(digest):
        raise ValueError('Verified campaign manifest identity required')
    return [dict(row, native_trial_id=row['trial_id'], trial_id=digest+':'+row['trial_id'],
                 campaign_manifest_sha256=digest) for row in verified['trials']]


def propose_revision(parent, patch, development, *, protocol=None):
    """Apply whitelisted JSON edits, never agent-supplied executable code."""
    validate_template(parent)
    p = validate_protocol(protocol)
    if (set(patch) != {'version', 'base_template_sha256', 'reason', 'evidence_ids', 'operations'}
            or patch['version'] != VERSION or patch['base_template_sha256'] != template_hash(parent)
            or not isinstance(patch['reason'], str) or not patch['reason']):
        raise ValueError('Invalid data-only revision proposal')
    checked = development_summary(development['rows'])
    if checked != development or checked['task'] != parent['task']:
        raise ValueError('Development ledger changed or task differs')
    if (not isinstance(patch['evidence_ids'], list) or not patch['evidence_ids']
            or not set(patch['evidence_ids']) <= {r['trial_id'] for r in development['rows']}):
        raise ValueError('Proposal must cite development evidence, never final holdout outcomes')
    if not isinstance(patch['operations'], list) or not patch['operations']:
        raise ValueError('A nonempty whitelisted revision is required')
    revised = copy.deepcopy(parent)
    for op in patch['operations']:
        if set(op) != {'op', 'rule'} or op['op'] not in ('add_rule', 'replace_rule'):
            raise ValueError('Only data-only add_rule/replace_rule operations are allowed')
        rule = copy.deepcopy(op['rule'])
        matches = [i for i, existing in enumerate(revised['rules']) if existing['id'] == rule.get('id')]
        if op['op'] == 'add_rule':
            if matches:
                raise ValueError('New rule ID already exists')
            revised['rules'].append(rule)
        else:
            if len(matches) != 1:
                raise ValueError('Replacement needs an existing unique rule ID')
            old = revised['rules'][matches[0]]
            if any(old.get(k) != rule.get(k) for k in ACTIVATION_FIELDS):
                raise ValueError('Existing activation/subjects cannot be changed to evade counterexamples')
            revised['rules'][matches[0]] = rule
    validate_template(revised)
    growth = len(revised['rules'])-len(parent['rules'])
    if len(revised['rules']) > p['maximum_rules'] or growth > p['maximum_rule_growth']:
        raise ValueError('Revision exceeds the declared rule-complexity budget')
    if template_hash(revised) == template_hash(parent):
        raise ValueError('Revision did not change the template')
    return dict(version=VERSION, state='revision_proposed', task=parent['task'], parent=parent,
                parent_template_sha256=template_hash(parent), template=revised,
                template_sha256=template_hash(revised), patch=patch, protocol=p,
                development_report_sha256=json_hash(development),
                complexity=dict(parent_rules=len(parent['rules']), rules=len(revised['rules']), added_rules=growth))


def gripper_phase_proposal(parent, development, *, protocol=None):
    """Propose reference-activated opening/holding rules from timing loopholes.

    The family is a hypothesis selector, not a claim of causality or approval.
    Independent held-out native runs still decide whether this revision works.
    """
    counterexamples = [r for r in development['rows'] if classify(r)['strong_false_accept']
                       and ('grip' in r['family'] or 'grip' in r.get('candidate_id', ''))]
    if not counterexamples:
        return None
    rules = []
    existing = {r['id'] for r in parent['rules']}
    for state in ('all',):
        identifier = 'rsi_gripper_'+state+'_reference_plateaus'
        if identifier in existing:
            continue
        rules.append(dict(id=identifier, type='gripper_interval', subjects='active_arms',
                          selector=dict(kind='gripper_plateaus', state=state, after='episode_start'),
                          origin='world', orientation='world', parameters={},
                          rationale='Development counterexample suggests an event-only gripper loophole; '
                                    'test reference-activated phase opening/holding on independent scenes.'))
    if not rules:
        return None
    patch = dict(version=VERSION, base_template_sha256=template_hash(parent),
                 reason='Test a gripper-phase hypothesis from verified constraint-pass/task-fail development cases.',
                 evidence_ids=[r['trial_id'] for r in counterexamples],
                 operations=[dict(op='add_rule', rule=r) for r in rules])
    return propose_revision(parent, patch, development, protocol=protocol)


def seal_revision(revision, split_plan, development, *, frozen_unix=None):
    validate_split_plan(split_plan)
    generator = validate_generator(split_plan.get('holdout_generator'))
    expected = propose_revision(revision['parent'], revision['patch'], development, protocol=revision['protocol'])
    if revision != expected or revision['task'] != split_plan['task']:
        raise ValueError('Revision changed before sealing')
    if any(r['scene_seed'] not in split_plan['dev_seeds'] for r in development['rows']):
        raise ValueError('Development ledger contains final holdout or undeclared seeds')
    frozen = time.time() if frozen_unix is None else frozen_unix
    if type(frozen) not in (float, int) or not math.isfinite(frozen) or frozen <= 0:
        raise ValueError('Invalid freeze time')
    sealed = dict(revision, state='sealed', split_plan_sha256=json_hash(split_plan), frozen_unix=frozen,
                  annotation_policy_sha256=annotation_policy_fingerprint())
    release = dict(version=VERSION, task=revision['task'], template_sha256=revision['template_sha256'],
                   annotation_policy_sha256=sealed['annotation_policy_sha256'],
                   split_plan_sha256=json_hash(split_plan), frozen_unix=frozen,
                   development_report_sha256=json_hash(development), revision_sha256=json_hash(sealed),
                   protocol_sha256=json_hash(revision['protocol']), holdout_generator_sha256=json_hash(generator))
    return sealed, release


def binomial_upper(failures, scenes, confidence):
    """One-sided exact Clopper-Pearson upper bound, computed without SciPy."""
    if not 0 <= failures <= scenes or scenes < 1 or not 0 < confidence < 1:
        raise ValueError('Invalid binomial bound inputs')
    if failures == scenes:
        return 1.
    if failures == 0:
        return -math.expm1(math.log1p(-confidence)/scenes)
    alpha = 1-confidence
    low, high = failures/scenes, 1.
    for _ in range(80):
        probability = (low+high)/2
        logs = [math.lgamma(scenes+1)-math.lgamma(k+1)-math.lgamma(scenes-k+1)
                + k*math.log(probability)+(scenes-k)*math.log1p(-probability)
                for k in range(failures+1)]
        top = max(logs)
        cdf = math.exp(top)*sum(math.exp(x-top) for x in logs)
        if cdf > alpha:
            low = probability
        else:
            high = probability
    return (low+high)/2


def assess_holdout(sealed, release, split_plan, development, rows, *, complete, preregistered_trial_ids):
    """Assess verified raw-evidence rows; production callers must use assess_revision."""
    proposal = {k: v for k, v in sealed.items()
                if k not in ('split_plan_sha256', 'frozen_unix', 'annotation_policy_sha256')}
    proposal['state'] = 'revision_proposed'
    expected_sealed, expected_release = seal_revision(proposal, split_plan, development,
                                                      frozen_unix=sealed['frozen_unix'])
    if sealed != expected_sealed or {k: v for k, v in release.items() if k != 'exposure_reservation'} != expected_release:
        raise ValueError('Frozen revision/release changed or implementation policy differs')
    p = validate_protocol(sealed['protocol'])
    reasons = []
    if (not complete or len(set(preregistered_trial_ids)) != len(preregistered_trial_ids)
            or len({r['trial_id'] for r in rows}) != len(rows)
            or set(preregistered_trial_ids) != {r['trial_id'] for r in rows}):
        reasons.append('holdout_plan_incomplete_or_duplicate')
    dev_scenes = {r['scene_sha256'] for r in development['rows']}
    identities = {}
    for row in rows:
        validate_ledger_row(row)
        if (row['split'] != 'holdout' or row['task'] != sealed['task']
                or row['scene_seed'] not in split_plan['holdout_seeds']
                or row['scene_sha256'] in dev_scenes
                or row['template_sha256'] != sealed['template_sha256']
                or row['annotation_policy_sha256'] != sealed['annotation_policy_sha256']):
            raise ValueError('Holdout row is contaminated or differs from the frozen revision')
        identities.setdefault(row['scene_seed'], set()).add(row['scene_sha256'])
    if (any(len(v) != 1 for v in identities.values())
            or len({next(iter(v)) for v in identities.values()}) != len(identities)):
        raise ValueError('Distinct holdout seeds must identify distinct immutable scenes')
    stats = {}
    generator = validate_generator(split_plan['holdout_generator'])
    sampling_complete = complete and not reasons
    for seed in split_plan['holdout_seeds']:
        baselines = [r for r in rows if r['scene_seed'] == seed and r.get('kind') == 'baseline']
        if (len(baselines) != generator['configuration']['baseline_repeats']
                or any(not (eligible(r) and r['native_success'] and r['commanded_constraint_pass']) for r in baselines)):
            reasons.append('holdout_native_baseline_unreliable')
            sampling_complete = False
    for source in SOURCES:
        selected = [r for r in rows if r['source'] == source and r.get('kind', 'candidate') == 'candidate']
        stats[source] = summary = source_summary(selected)
        n, k = summary['accepted_scenes'], summary['failed_accepted_scenes']
        # No adaptive discovery rows enter this computation. Within-scene
        # repeated/correlated variants collapse to one worst-case indicator.
        summary['scene_failure_upper_bound'] = binomial_upper(k, n, p['confidence']) if n else None
        summary['confidence_eligible'] = False
        if source in p['required_sources']:
            budgets = {seed: sum(r['scene_seed'] == seed for r in selected) for seed in split_plan['holdout_seeds']}
            if any(count != generator['sources'][source]['candidates_per_scene'] for count in budgets.values()):
                reasons.append(source+'_preregistered_scene_budget_incomplete')
                sampling_complete = False
            if source == 'model' and any(
                    any(r['model_provenance'][key] != generator['sources'][source][key]
                        for key in ('checkpoint_sha256', 'source_kind')) for r in selected):
                raise ValueError('Model source changed after the holdout generator was frozen')
            if source == 'model' and any(
                    r['model_provenance'].get('candidate_is_full_policy_rollout') is not True
                    or r['model_provenance'].get('search_transform') != 'none'
                    or r['model_provenance'].get('reference_blend') != 1 for r in selected):
                raise ValueError('Transformed model candidates cannot occupy actual full-policy holdout slots')
            if n < p['minimum_accepted_scenes']:
                reasons.append(source+'_insufficient_independent_accepted_scenes')
            if summary['native_success_scenes'] < p['minimum_native_success_scenes']:
                reasons.append(source+'_insufficient_success_diversity')
            retention = summary['scene_normalized_success_retention']
            if retention is None or retention < p['minimum_success_retention']:
                reasons.append(source+'_rejects_successful_behaviors')
            execution = summary['minimum_scene_accepted_execution_fraction']
            if execution is None or execution < p['minimum_accepted_execution_fraction']:
                reasons.append(source+'_accepted_execution_coverage_insufficient')
                sampling_complete = False
            if n and summary['scene_failure_upper_bound'] > p['maximum_scene_failure_risk']:
                reasons.append(source+'_risk_bound_not_met')
            if p['require_zero_observed_false_accepts'] and summary['false_accepts']:
                reasons.append(source+'_observed_accepted_failures')
    if len({r['family'] for r in rows if r['source'] == 'crafted' and eligible(r)}) < p['minimum_crafted_families']:
        reasons.append('insufficient_crafted_behavior_diversity')
    for source in p['required_sources']:
        stats[source]['confidence_eligible'] = sampling_complete and stats[source]['accepted_scenes'] >= p['minimum_accepted_scenes']
        if not stats[source]['confidence_eligible']:
            stats[source]['scene_failure_upper_bound'] = None
    return dict(version=VERSION, state='accepted' if not reasons else 'needs_revision', ready=not reasons,
                task=sealed['task'], template_sha256=sealed['template_sha256'],
                annotation_policy_sha256=sealed['annotation_policy_sha256'], protocol=p,
                release_sha256=json_hash(release), records_sha256=json_hash(rows),
                reasons=sorted(set(reasons)), sources=stats,
                risk_unit='one independently seeded scene per source; any accepted failure is a scene failure',
                meaning='Conditional scene-level risk for the frozen declared generator and policy sources only; '
                        'not an all-actions success guarantee or an IID claim about adaptive discovery variants.')


def assess_revision(revision_root, *, verify_campaign=None, expected_registry=None):
    """Re-read frozen inputs and independently verify every native campaign."""
    root = Path(revision_root).resolve()
    if verify_campaign is None:
        from scripts.search_robotwin_constraint_counterexamples import verify_campaign
    sealed, release, split_plan, development = [read(root/name) for name in (
        'sealed_revision.json', 'holdout_release.json', 'split_plan.json', 'development_report.json')]
    inventory = read(root/'holdout_campaigns.json')
    if set(inventory) != {'version', 'campaigns'} or inventory['version'] != VERSION or not inventory['campaigns']:
        raise ValueError('Explicit frozen holdout campaign inventory required')
    rows, planned, proofs = [], [], []
    complete = True
    for path in inventory['campaigns']:
        checked = verify_campaign(Path(path))
        manifest = checked['manifest']
        if (manifest.get('split') != 'holdout' or manifest.get('task') != sealed['task']
                or manifest.get('template_sha256') != sealed['template_sha256']
                or manifest.get('annotation_policy_sha256') != sealed['annotation_policy_sha256']
                or manifest.get('split_plan_sha256') != json_hash(split_plan)
                or manifest.get('holdout_release_sha256') != json_hash(release)
                or manifest.get('holdout_generator_sha256') != release['holdout_generator_sha256']
                or manifest.get('generator_sha256') != split_plan['holdout_generator']['generator_sha256']
                or manifest.get('feedback_used') is not False
                or not isinstance(manifest.get('created_unix'), (int, float))
                or manifest['created_unix'] < release['frozen_unix']):
            raise ValueError('Campaign differs from the sealed independent holdout plan')
        configuration = read(Path(path)/'configuration.json')
        if any(configuration.get(k) != v for k, v in split_plan['holdout_generator']['configuration'].items()):
            raise ValueError('Campaign generation parameters differ from the sealed holdout distribution')
        rows.extend(campaign_rows(checked))
        planned.extend(checked['manifest_sha256']+':'+item['trial_id'] for item in manifest['trial_plan'])
        complete &= checked['complete']
        proofs.append(dict(path=str(Path(path).resolve()), manifest_sha256=checked['manifest_sha256'],
                           records_sha256=checked['records_sha256']))
    report = assess_holdout(sealed, release, split_plan, development, rows, complete=complete,
                            preregistered_trial_ids=planned)
    from services.robotwin.refinement_exposure import verify_reservation
    from services.robotwin.refinement_model_evidence import verify_model_evidence
    report['exposure'] = verify_reservation(root, release, rows, expected_registry=expected_registry)
    report['model_evidence'] = verify_model_evidence(root/'model_evidence.json', rows, release)
    report['evidence'] = dict(revision_root=str(root), campaigns=proofs,
                             frozen_inputs={name: file_hash(root/name) for name in (
                                 'sealed_revision.json', 'holdout_release.json', 'split_plan.json',
                                 'development_report.json', 'holdout_campaigns.json', 'model_evidence.json')})
    return report


def attach_revision(registry_root, revision_root):
    report = assess_revision(revision_root, expected_registry=registry_root)
    path = Path(registry_root)/'refinement_reports'/report['task']/(report['template_sha256']+'.json')
    atomic_json(path, report)
    return report


def refinement_readiness(registry_root, required_templates):
    root = Path(registry_root).resolve()
    requirement_path = root/'refinement_requirement.json'
    if not requirement_path.exists():
        return dict(required=False, ready=True, templates=[])
    requirement = read(requirement_path)
    if set(requirement) != {'version', 'protocol'} or requirement['version'] != VERSION:
        raise ValueError('Invalid required refinement gate')
    protocol = validate_protocol(requirement['protocol'])
    rows = []
    for task, hashes in sorted(required_templates.items()):
        for digest in sorted(set(hashes)):
            path = root/'refinement_reports'/task/(digest+'.json')
            if not path.exists():
                rows.append(dict(task=task, template_sha256=digest, ready=False,
                                 reason='independent_refinement_evidence_missing'))
                continue
            report = read(path)
            fresh = assess_revision(report['evidence']['revision_root'], expected_registry=root)
            ready = (derived_values_equal(fresh, report) and report['ready'] and report['task'] == task
                     and report['template_sha256'] == digest and report['protocol'] == protocol)
            rows.append(dict(task=task, template_sha256=digest, ready=ready, report_path=str(path),
                             report_sha256=file_hash(path), reason=None if ready else 'refinement_evidence_not_ready'))
    return dict(required=True, ready=bool(rows) and all(r['ready'] for r in rows),
                requirement=requirement, requirement_sha256=file_hash(requirement_path),
                templates=rows)


def validate_frozen_gate(gate, *, required_templates=None, required=False):
    """Dataset training validation retains the stronger gate after publication."""
    if not gate.get('required'):
        if required:
            raise ValueError('Independent constraint refinement evidence is required for this training')
        return
    if not gate.get('ready') or not gate.get('templates'):
        raise ValueError('Refinement gate was not ready at publication')
    protocol = validate_protocol(gate['requirement']['protocol'])
    if required_templates is not None:
        expected = {(task, digest) for task, hashes in required_templates.items() for digest in hashes}
        actual = {(row['task'], row['template_sha256']) for row in gate['templates']}
        if actual != expected or len(actual) != len(gate['templates']):
            raise ValueError('Refinement proof must cover every published template exactly once')
    for row in gate['templates']:
        path = Path(row['report_path'])
        if file_hash(path) != row['report_sha256']:
            raise ValueError('Frozen refinement report changed')
        report = read(path)
        if (not row['ready'] or not report['ready'] or report['protocol'] != protocol
                or report['task'] != row['task'] or report['template_sha256'] != row['template_sha256']
                or not derived_values_equal(assess_revision(report['evidence']['revision_root']), report)):
            raise ValueError('Frozen refinement evidence is not valid for training')
