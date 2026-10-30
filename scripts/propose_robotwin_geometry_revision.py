#!/usr/bin/env python3
"""Screen geometric hypotheses on immutable native development evidence only.

This does not run physics, produce new successes, or activate a template. Old
native outcomes are reverified; only their reference-fixed labels are rescored.
Use --runtime-root to import a frozen CPU code snapshot without modifying it.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys


def geometry_rule(kind, axis, margin, *, identifier=None):
    angle = kind == 'axis_angle_interval'
    label = ('angle' if angle else 'radial')+'_'+''.join(str(v) for v in axis)
    return dict(id=identifier or 'dev_geometry_'+label, type=kind, subjects='active_arms',
                selector=dict(kind='gripper_plateaus', state='closed', after='episode_start'),
                origin='selection_start', orientation='selection_start' if angle else 'world',
                parameters=(dict(local_axis=axis, reference_axis=axis) if angle else dict(axis=axis)),
                allowed_margin=dict(lower=margin, upper=margin, unit='rad' if angle else 'm'),
                rationale=('Development hypothesis: retain one tool-axis alignment relative to the '
                           'reference plateau-start frame; free rotation about that axis remains allowed. '
                           if angle else 'Development hypothesis: retain a lateral tube around the reference '
                           'plateau-start world-Z axis while allowing upward displacement. ')
                          +'The axes and physical margins are uncalibrated proposals, not contact or grasp proof.')


def proposal_grid():
    """Finite development grid declared before reading outcomes; no scene tuning."""
    rows = [('gripper_025_only', [])]
    axes = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    for index, axis in enumerate(axes):
        for degrees in (5, 10, 15):
            rows.append((f'axis_{index}_angle_{degrees}', [geometry_rule(
                'axis_angle_interval', axis, math.radians(degrees))]))
    for millimeters in (5, 10):
        rows.append((f'radial_z_{millimeters}mm', [geometry_rule(
            'radial_distance_interval', [0, 0, 1], millimeters/1000)]))
    for index, axis in enumerate(axes):
        for degrees in (5, 10):
            rows.append((f'axis_{index}_angle_{degrees}_radial_z_5mm', [
                geometry_rule('axis_angle_interval', axis, math.radians(degrees)),
                geometry_rule('radial_distance_interval', [0, 0, 1], .005)]))
    for degrees in (5, 10):
        rows.append((f'axes_xz_angle_{degrees}', [geometry_rule(
            'axis_angle_interval', axis, math.radians(degrees)) for axis in (axes[0], axes[2])]))
    return rows


def screen(export, ledger_path, output, *, task_name='lift_pot', relax_existing=False):
    from scripts.validate_robotwin_constraint_acceptance import reference_bundle
    from services.robotwin.acceptance import (
        _artifact, annotation_policy_fingerprint, evaluate_trial, file_hash, json_hash, trajectory_violations,
    )
    from services.robotwin.collection import atomic_json
    from services.robotwin.constraint_refinement import (
        classify, development_summary, gripper_phase_proposal, propose_revision, source_summary,
    )

    output = Path(output)
    if output.exists():
        raise ValueError('Use a new output directory; old native evidence/proposals are immutable')
    document = json.loads(Path(export).read_text())
    tasks = [item for item in document['tasks'] if item['task'] == task_name]
    if len(tasks) != 1:
        raise ValueError('Exactly one task export required')
    task = tasks[0]
    manifest = json.loads(Path(task['manifest_path']).read_text())
    if manifest != task['source_manifest']:
        raise ValueError('Source native manifest changed')
    development = json.loads(Path(ledger_path).read_text())
    if development_summary(development['rows']) != development or development['task'] != task_name:
        raise ValueError('Development ledger changed or task mismatches')
    ledger = development['rows']
    exports = {item['trial_id']: item for item in task['rows']}
    if set(exports) != {item['trial_id'] for item in ledger}:
        raise ValueError('Screen all preregistered historical trials; no outcome filtering')
    traces, receipts = {}, []
    for row in ledger:
        evidence = row['origin_evidence']
        root, record = Path(evidence['root']), evidence['record']
        original = exports[row['trial_id']]
        if record != original['record'] or json_hash(record) != row['evidence_sha256']:
            raise ValueError('Development row evidence differs from frozen original')
        checked = evaluate_trial(record, root, manifest)
        reference = _artifact(root, record, 'reference')
        trace = _artifact(root, record, 'trajectory')
        outcome = _artifact(root, record, 'outcome')
        if (checked['eligible'] != classify(row)['eligible']
                or checked['native_success'] != row['native_success']
                or checked.get('constraint_pass') != row['actual_constraint_pass']
                or checked['eligible'] != original['eligible']
                or row['trajectory_sha256'] != json_hash(trace['commanded'])
                or row['actual_trajectory_sha256'] != json_hash(trace['actual'])
                or outcome['execution_complete'] != row['execution_complete']):
            raise ValueError('Historical native trajectory/outcome differs from reverified evidence')
        if outcome['execution_complete']:
            command = trajectory_violations(reference['bundle'], reference['trajectory'], trace['commanded'])
            actual = trajectory_violations(reference['bundle'], reference['trajectory'], trace['actual'])
            if (row['commanded_constraint_pass'] != command['passed']
                    or (row['actual_constraint_pass'] is not None and row['actual_constraint_pass'] != actual['passed'])):
                raise ValueError('Historical constraint labels differ from original bundle')
        traces[row['trial_id']] = reference, trace
        receipts.append(dict(trial_id=row['trial_id'], record_sha256=json_hash(record),
                             reference_sha256=record['reference_sha256'],
                             trajectory_sha256=record['trajectory_sha256'], outcome_sha256=record['outcome_sha256']))
    base = gripper_phase_proposal(task['template'], development)
    if base is None:
        raise ValueError('No verified gripper hypothesis in this development ledger')
    grid = proposal_grid()
    relaxations = {}
    if relax_existing:
        if task_name != 'lift_pot':
            raise ValueError('The existing-margin hypothesis is specific to the reviewed lift_pot template')
        grid = []
        for height_margin in (.005, .015):
            for closed_margin in (.02, .25):
                name = f'axis_x10_height_{height_margin:g}_closed_{closed_margin:g}'
                grid.append((name, [geometry_rule('axis_angle_interval', [1, 0, 0], math.radians(10))]))
                relaxations[name] = dict(terminal_height=dict(lower=height_margin, upper=height_margin, unit='m'),
                                         fully_closed_command=dict(lower=closed_margin, upper=closed_margin, unit='fraction'))
    configuration = dict(version=1, task=task_name, development_only=True,
                         source_export_sha256=file_hash(export), source_ledger_sha256=file_hash(ledger_path),
                         script_sha256=file_hash(__file__), annotation_policy_sha256=annotation_policy_fingerprint(),
                         gripper_margin_fraction=.25,
                         existing_rule_margins=relaxations,
                         geometric_grid=[dict(name=name, rules=rules) for name, rules in grid])
    atomic_json(output/'configuration.json', configuration)
    atomic_json(output/'verified_source_receipts.json', receipts)
    atomic_json(output/'source_development_report.json', development)
    groups = {}
    for row in ledger:
        if classify(row)['eligible']:
            groups.setdefault((row['scene_sha256'], row['trajectory_sha256']), []).append(row)
    duplicate_groups = [[r['trial_id'] for r in rows] for rows in groups.values() if len(rows) > 1]
    mixed_groups = [[r['trial_id'] for r in rows] for rows in groups.values()
                    if len({r['native_success'] for r in rows}) > 1]
    summary = dict(version=1, task=task_name, state='development_hypotheses_only',
                   native_replay_performed=False, acceptance_ready=False, confidence_eligible=False,
                   source=source_summary(ledger),
                   source_candidates_only=source_summary([r for r in ledger if r['kind'] == 'candidate']),
                   unique_eligible_scene_action_pairs=len(groups), duplicate_scene_action_groups=duplicate_groups,
                   indistinguishable_action_mixed_outcome_groups=mixed_groups, proposals=[])
    for name, added_rules in grid:
        patch = copy.deepcopy(base['patch'])
        patch['reason'] = ('Screen fixed reference-activated gripper and minimal geometry combinations on '
                           'already observed native development outcomes; no acceptance claim.')
        patch['evidence_ids'] = [row['trial_id'] for row in ledger if classify(row)['strong_false_accept']]
        patch['operations'][0]['rule']['allowed_margin'] = dict(lower=.25, upper=.25, unit='fraction')
        patch['operations'].extend(dict(op='add_rule', rule=rule) for rule in added_rules)
        for identifier, margin in relaxations.get(name, {}).items():
            matching = [rule for rule in task['template']['rules'] if rule['id'] == identifier]
            if len(matching) != 1:
                raise ValueError('Reviewed rule to relax is missing or ambiguous')
            replacement = copy.deepcopy(matching[0])
            replacement['allowed_margin'] = margin
            patch['operations'].append(dict(op='replace_rule', rule=replacement))
        revision = propose_revision(task['template'], patch, development)
        revised_rows, verdicts = [], []
        for old in ledger:
            reference, trajectory = traces[old['trial_id']]
            revised = copy.deepcopy(old)
            revised['template_sha256'] = revision['template_sha256']
            revised['annotation_policy_sha256'] = annotation_policy_fingerprint()
            detail = dict(trial_id=old['trial_id'], eligible=classify(old)['eligible'],
                          native_success=old['native_success'], before=classify(old))
            if old['execution_complete']:
                bundle, audit = reference_bundle(revision['template'], reference['trajectory'],
                                                  old['scene_seed'], old['scene_sha256'])
                baseline = trajectory_violations(bundle, reference['trajectory'], reference['trajectory'])
                if not baseline['passed']:
                    raise ValueError('Hypothesis unexpectedly rejects its reference')
                command = trajectory_violations(bundle, reference['trajectory'], trajectory['commanded'])
                actual = trajectory_violations(bundle, reference['trajectory'], trajectory['actual'])
                revised['commanded_constraint_pass'], revised['actual_constraint_pass'] = command['passed'], actual['passed']
                detail.update(commanded=command, actual=actual, bundle_sha256=json_hash(bundle),
                              template_audit_sha256=json_hash(audit))
            detail['after'] = classify(revised)
            revised_rows.append(revised)
            verdicts.append(detail)
        report = dict(version=1, name=name, state='proposal_not_accepted', native_replay_performed=False,
                      confidence_eligible=False, acceptance_ready=False,
                      source=summary['source'], revised=source_summary(revised_rows),
                      revised_candidates_only=source_summary([r for r in revised_rows if r['kind'] == 'candidate']),
                      template_sha256=revision['template_sha256'],
                      newly_rejected_success_ids=[a['trial_id'] for a, b in zip(ledger, revised_rows)
                          if classify(a)['eligible'] and a['native_success']
                          and a['commanded_constraint_pass'] and not b['commanded_constraint_pass']],
                      all_rejected_success_ids=[r['trial_id'] for r in revised_rows if classify(r)['false_reject']],
                      rejected_known_false_accepts=[a['trial_id'] for a, b in zip(ledger, revised_rows)
                          if classify(a)['false_accept'] and not classify(b)['false_accept']],
                      remaining_false_accepts=[r['trial_id'] for r in revised_rows if classify(r)['false_accept']],
                      rows=verdicts,
                      limitations=['Same development scenes used to screen a finite hypothesis grid.',
                                   'No new native replay; immutable historical outcomes do not improve.',
                                   'Source trajectories are expert perturbations, not actual model behavior.',
                                   'Single-axis alignment retains twist freedom; bimanual relative geometry is absent.',
                                   'Passing this screen cannot activate a revision or certify tolerance.'])
        atomic_json(output/name/'revision_proposal.json', revision)
        atomic_json(output/name/'rescore_report.json', report)
        summary['proposals'].append({key: value for key, value in report.items() if key not in ('rows', 'limitations')})
    atomic_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='lift_pot')
    parser.add_argument('--relax-existing', action='store_true',
                        help='Screen the four preregistered height/closed-gripper margin combinations for lift_pot')
    parser.add_argument('--runtime-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime_root.resolve()))
    result = screen(args.export, args.ledger, args.output, task_name=args.task, relax_existing=args.relax_existing)
    print(json.dumps(dict(task=result['task'], source=result['source'], proposals=[dict(
        name=p['name'], accepted_successes=p['revised']['native_successes']-p['revised']['false_rejects'],
        false_accepts=p['revised']['false_accepts'], newly_rejected_success_ids=p['newly_rejected_success_ids'])
        for p in result['proposals']]), indent=2))


if __name__ == '__main__':
    main()
