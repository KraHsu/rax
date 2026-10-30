#!/usr/bin/env python3
"""Read-only native-evidence rescore; write development proposals, never activate.

Historical simulator outcomes remain immutable. Every source artifact is hashed
and execution rechecked before testing added reference-plateau constraints.
This is offline hypothesis screening, not new replay or holdout acceptance.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_robotwin_constraint_acceptance import reference_bundle  # noqa: E402
from services.robotwin.acceptance import (  # noqa: E402
    _artifact, annotation_policy_fingerprint, evaluate_trial, file_hash, json_hash,
    trajectory_violations,
)
from services.robotwin.collection import atomic_json  # noqa: E402
from services.robotwin.constraint_refinement import (  # noqa: E402
    classify, development_summary, gripper_phase_proposal, propose_revision, source_summary,
)


def rescore(export, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Use a new proposal directory; previous evidence must not be overwritten')
    document = json.loads(Path(export).read_text())
    summary = dict(version=1, state='development_proposals_only', acceptance_ready=False,
                   confidence_eligible=False, source_export_sha256=file_hash(export),
                   annotation_policy_sha256=annotation_policy_fingerprint(), tasks=[])
    for task in document['tasks']:
        manifest = json.loads(Path(task['manifest_path']).read_text())
        if manifest != task['source_manifest']:
            raise ValueError('Exported manifest differs from its frozen native source')
        ledger, traces = [], {}
        for original in task['rows']:
            record, root = original['record'], Path(original['evidence_root'])
            checked = evaluate_trial(record, root, manifest)
            reference = _artifact(root, record, 'reference')
            trajectory = _artifact(root, record, 'trajectory')
            outcome = _artifact(root, record, 'outcome')
            if checked['eligible'] != original['eligible'] or checked['native_success'] != original['native_success']:
                raise ValueError('Exported outcome differs from reverified native evidence')
            command_score = (trajectory_violations(reference['bundle'], reference['trajectory'], trajectory['commanded'])
                             if outcome['execution_complete'] else None)
            actual_pass = checked.get('constraint_pass')
            row = dict(trial_id=record['trial_id'], task=record['task'], split='dev',
                       scene_seed=record['seed'], scene_sha256=record['scene_sha256'],
                       source='expert', origin_source='expert_perturbation', family=record['variant_family'],
                       kind='baseline' if record['variant_family'] == 'baseline' else 'candidate',
                       native_success=outcome.get('native_success'), execution_complete=outcome['execution_complete'],
                       planner_success=outcome['planner_success'],
                       realized=bool(outcome['realized'] and checked['eligible']),
                       recorded_tracking_realized=outcome['realized'],
                       commanded_constraint_pass=bool(command_score and command_score['passed']),
                       actual_constraint_pass=actual_pass,
                       trajectory_sha256=json_hash(trajectory['commanded']),
                       actual_trajectory_sha256=json_hash(trajectory['actual']),
                       reference_sha256=record['reference_sha256'], template_sha256=record['template_sha256'],
                       annotation_policy_sha256=record['annotation_policy_sha256'], evidence_sha256=json_hash(record),
                       origin_evidence=dict(root=str(root), record=record, verified_by='evaluate_trial'))
            ledger.append(row)
            traces[row['trial_id']] = (reference, trajectory)
        dev = development_summary(ledger)
        atomic_json(output/task['task']/'development_report.json', dev)
        original_summary = source_summary(ledger)
        item = dict(task=task['task'], source=original_summary, proposals=[])
        proposal = gripper_phase_proposal(task['template'], dev)
        if proposal:
            for margin in (.02, .1, .25, .49):
                patch = copy.deepcopy(proposal['patch'])
                patch['operations'][0]['rule']['allowed_margin'] = dict(lower=margin, upper=margin, unit='fraction')
                revision = propose_revision(task['template'], patch, dev)
                revised_rows, verdicts = [], []
                for old in ledger:
                    reference, trajectory = traces[old['trial_id']]
                    revised = copy.deepcopy(old)
                    revised['template_sha256'] = revision['template_sha256']
                    revised['annotation_policy_sha256'] = annotation_policy_fingerprint()
                    detail = dict(trial_id=old['trial_id'], eligible=classify(old)['eligible'],
                                  native_success=old['native_success'], before=classify(old))
                    if old['execution_complete']:
                        bundle, _ = reference_bundle(revision['template'], reference['trajectory'],
                                                     old['scene_seed'], old['scene_sha256'])
                        command_score = trajectory_violations(bundle, reference['trajectory'], trajectory['commanded'])
                        actual_score = trajectory_violations(bundle, reference['trajectory'], trajectory['actual'])
                        revised['commanded_constraint_pass'] = command_score['passed']
                        revised['actual_constraint_pass'] = actual_score['passed']
                        detail.update(commanded=command_score, actual=actual_score)
                    detail['after'] = classify(revised)
                    revised_rows.append(revised); verdicts.append(detail)
                after = source_summary(revised_rows)
                path = output/task['task']/f'gripper_plateau_margin_{margin:g}'
                report = dict(version=1, state='proposal_not_accepted', native_replay_performed=False,
                              confidence_eligible=False, acceptance_ready=False,
                              allowed_margin_fraction=margin, source=original_summary, revised=after,
                              formerly_accepted_successes=sum(classify(r)['eligible'] and r['native_success']
                                                              and r['commanded_constraint_pass'] for r in ledger),
                              newly_rejected_success_ids=[a['trial_id'] for a, b in zip(ledger, revised_rows)
                                                          if classify(a)['eligible'] and a['native_success']
                                                          and a['commanded_constraint_pass'] and not b['commanded_constraint_pass']],
                              rejected_known_false_accepts=[a['trial_id'] for a, b in zip(ledger, revised_rows)
                                                           if classify(a)['false_accept'] and not classify(b)['false_accept']],
                              remaining_false_accepts=[r['trial_id'] for r in revised_rows if classify(r)['false_accept']],
                              rows=verdicts,
                              limitations=['Historical expert perturbations only; no actual model evidence.',
                                           'Same development scenes reused for screening, never final holdout.',
                                           'Gripper phase hypothesis does not fix yaw or coupled position/rotation holes.',
                                           'Successful source outcome unchanged; revised constraint labels rescored offline.'])
                atomic_json(path/'revision_proposal.json', revision)
                atomic_json(path/'rescore_report.json', report)
                item['proposals'].append({k: v for k, v in report.items() if k not in ('rows', 'limitations')})
        summary['tasks'].append(item)
    atomic_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rescore(args.export, args.output), indent=2))
