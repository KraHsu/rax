"""Audited development observations, deliberately outside formal acceptance.

An executed trajectory is useful constraint evidence even when a proposed
intervention was not tracked. The original strict classification is retained;
this module never changes its tolerances or issues acceptance certificates.
"""
import copy
import json
from pathlib import Path

from services.robotwin.acceptance import (
    annotation_policy_fingerprint, derived_values_equal, evaluator_fingerprint, file_hash, json_hash,
)
from services.robotwin.acceptance_replay import ALIGNMENT, SUCCESS_RULE, TRACKING, active_arms, intervene
from services.robotwin.adversarial_search import (
    SEARCH_VERSION, SELECTION_POLICY, score_candidate, search_manifest,
    source_fingerprint,
)
from scripts.search_robotwin_constraint_counterexamples import (
    classify_trial, prepare_reference, read,
)

KIND = 'observed_trajectory_discovery_dev_only'
PURPOSE = 'research_action_compliance_not_formal_acceptance'
MANIFEST = 'observed_manifest.json'


def runtime_fingerprint():
    root = Path(__file__).resolve().parents[2]
    return {name: file_hash(root/name) for name in (
        'services/robotwin/observed_discovery.py', 'services/robotwin/acceptance_replay.py',
        'services/robotwin/collection.py', 'services/robotwin/runtime.py',
        'services/robotwin/success_metadata.py', 'services/robotwin/instruction_metadata.py',
        'scripts/discover_robotwin_observed_constraints.py',
        'scripts/search_robotwin_constraint_counterexamples.py')}


def contained(root, relative):
    root = Path(root).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root) or path == root or Path(relative).is_absolute():
        raise ValueError('Observed evidence paths must stay inside the frozen package')
    return path


def checked(root, spec):
    path = contained(root, spec['path'])
    if file_hash(path) != spec['sha256']:
        raise ValueError('Observed artifact hash changed')
    return read(path)


def validate_design(root, design):
    """Finite first campaign: one exposed microwave scene, 20+4 fixed proposals."""
    if (design.get('version') != 1 or design.get('kind') != KIND
            or design.get('purpose') != PURPOSE or design.get('split') != 'dev'
            or design.get('task') != 'open_microwave' or design.get('seed') != 10000):
        raise ValueError('Only the reviewed microwave development design is supported')
    for field in ('confidence_eligible', 'holdout_allowed', 'publication_allowed',
                  'acceptance_ready', 'reference_recapture', 'scene_replacement', 'adaptive_feedback'):
        if design.get(field) is not False:
            raise ValueError('Observed discovery cannot become formal evidence or adapt this round')
    if (design.get('automatic_retries') != 0 or design.get('total_native_slots_max') != 25
            or design.get('generator_budget') != 20 or design.get('unfiltered_signed_control_budget') != 4
            or design.get('generator_seed') != 20261030 or design.get('strict_tracking') != TRACKING):
        raise ValueError('Finite budget, seed or unchanged strict tracking contract differs')
    if (design.get('candidate_validation') != 'sealed_artifacts_and_recomputed_scores_no_regeneration'
            or design.get('generation_numeric_context', {}).get('numpy_version') != '2.5.4'):
        raise ValueError('Frozen generation context and immutable-candidate validation required')
    expected_control = dict(execution_complete=True, planner_success=True, native_success=True,
                            strict_tracking_required=False, constraint_pass_required=False,
                            initial_state_and_source_verified=True)
    if design.get('control_requirement') != expected_control:
        raise ValueError('Observed control requirement changed')
    if (design.get('annotation_policy_sha256') != annotation_policy_fingerprint()
            or design.get('evaluator_sha256') != evaluator_fingerprint()
            or design.get('generator_sha256') != source_fingerprint()
            or design.get('generator_version') != SEARCH_VERSION
            or design.get('selection_policy') != SELECTION_POLICY):
        raise ValueError('Frozen scoring or proposal implementation changed')
    files = design['source_files']
    required = {'inputs/source_reference.json', 'inputs/source_manifest.json', 'inputs/reference.json',
                'inputs/template.json', 'inputs/training_exclusions.json', 'inputs/selection_audit.json',
                *{f'inputs/candidate_{i:04d}.json' for i in range(24)}}
    if set(files) != required:
        raise ValueError('Observed design must seal exactly its reviewed data inputs')
    for name, digest in files.items():
        if file_hash(contained(root, name)) != digest:
            raise ValueError('Frozen observed input changed: '+name)
    source, manifest, reference, template, exclusions = (
        read(contained(root, 'inputs/'+name+'.json'))
        for name in ('source_reference', 'source_manifest', 'reference', 'template', 'training_exclusions'))
    if (files['inputs/source_reference.json'] != design['source_reference_sha256']
            or files['inputs/source_manifest.json'] != design['source_manifest_sha256']):
        raise ValueError('Source evidence identity changed')
    matches = [r for r in manifest['reference_artifacts'] if r['seed'] == design['seed']]
    if len(matches) != 1 or matches[0]['sha256'] != design['source_reference_sha256']:
        raise ValueError('Source reference is not the exact original artifact')
    if (manifest['native_commit'] != design['native_commit']
            or source['provenance']['seed'] != design['seed']
            or source['provenance']['scene_sha256'] != design['scene_sha256']):
        raise ValueError('Observed task, seed, scene or native source differs')
    regenerated = prepare_reference(source, manifest, template, exclusions)
    if json_hash(regenerated) != json_hash(reference) or active_arms(reference['trajectory']) != ['left']:
        raise ValueError('Working labels changed or reference active arm differs')
    audit = read(contained(root, 'inputs/selection_audit.json'))
    candidates = []
    for i in range(24):
        candidate = read(contained(root, f'inputs/candidate_{i:04d}.json'))
        fresh_score = score_candidate(reference['trajectory'], reference['bundle'], candidate['trajectory'])
        if (candidate['commanded_constraint_pass'] is not fresh_score['commanded_constraint_pass']
                or not derived_values_equal(candidate['score'], fresh_score)):
            raise ValueError('Frozen candidate physical score changed')
        if i < 20:
            if candidate.get('observed_candidate_origin') != 'frozen_v2_selection' or candidate['source'] not in ('expert', 'crafted'):
                raise ValueError('Frozen proposal provenance is not the reviewed non-model selection')
        else:
            axis, sign = ((1, -1), (1, 1), (2, -1), (2, 1))[i-20]
            case = dict(id=f'outside_axis_{axis}_{sign}', variant_family='position',
                        position_axis=axis, position_offset_m=sign*.01)
            trajectory = intervene(reference['trajectory'], case)
            identity = dict(version='observed_fixed_controls_v1', source='crafted',
                            family='signed_position_control',
                            parameters=dict(case=case, reference_blend=1., active_arms=['left']),
                            reference_sha256=json_hash(reference['trajectory']),
                            bundle_sha256=json_hash(reference['bundle']), trajectory_sha256=json_hash(trajectory))
            expected = dict(identity, candidate_id='obs_'+json_hash(identity)[:20], trajectory=trajectory,
                            score=score_candidate(reference['trajectory'], reference['bundle'], trajectory),
                            commanded_constraint_pass=False,
                            observed_candidate_origin='fixed_unfiltered_position_control')
            # Frozen numeric trajectories stay byte-exact. Existing evaluator
            # tolerance covers arithmetic-only score differences across NumPy.
            expected['score'] = candidate['score']
            if json_hash(candidate) != json_hash(expected):
                raise ValueError('Unfiltered signed position control changed')
        candidates.append(candidate)
    search_manifest(reference['trajectory'], reference['bundle'], seed=20261030, budget=20,
                    candidates=candidates[:20], selection_audit=audit)
    generation = design.get('generation_exact_receipt', {})
    if (generation.get('generation_numpy') != design['generation_numeric_context']['numpy_version']
            or generation.get('source_fingerprint') != source_fingerprint()
            or generation.get('reference_trajectory_sha256') != json_hash(reference['trajectory'])
            or generation.get('bundle_sha256') != json_hash(reference['bundle'])
            or generation.get('exact_selection_audit_match') is not True
            or generation.get('exact_candidates') != 20
            or generation.get('candidate_trajectory_sha256') != [c['trajectory_sha256'] for c in candidates[:20]]):
        raise ValueError('Generation-context exact audit is not bound to these frozen candidates')
    if (audit.get('budget') != 20 or audit.get('selected') != 20
            or audit.get('reference_active_arms') != ['left'] or audit.get('policy') != SELECTION_POLICY
            or [r['candidate_id'] for r in audit['selected_candidates']] != [c['candidate_id'] for c in candidates[:20]]):
        raise ValueError('Frozen selection audit differs from the exact selected candidates')
    expected_plan = [dict(trial_id='control_0', kind='control', source='expert', family='reference_command',
                          trajectory_sha256=json_hash(reference['trajectory']))]
    for i, candidate in enumerate(candidates):
        expected_plan.append(dict(trial_id=f'candidate_{i:04d}', kind='candidate',
                                  candidate_path=f'inputs/candidate_{i:04d}.json',
                                  candidate_sha256=files[f'inputs/candidate_{i:04d}.json'],
                                  trajectory_sha256=candidate['trajectory_sha256'], source=candidate['source'],
                                  family=candidate['family'], candidate_id=candidate['candidate_id']))
    if design['trial_plan'] != expected_plan:
        raise ValueError('Observed trial slots changed, expanded or filtered')
    return reference, candidates, exclusions


def classify_observed(reference, candidate, raw, actual, *, control=None):
    strict = classify_trial(reference, candidate, raw, actual, baseline=control)
    observable = bool(strict['execution_complete'] and strict['actual_score'] is not None)
    return dict(strict=strict, strict_eligible=strict['eligible'], formal_evidence_eligible=False,
                strict_classification_scope='per_trial_only_not_campaign_acceptance',
                observationally_valid=observable,
                control_execution_reliable=bool(observable and strict['planner_success'] and strict['native_success']),
                observed_actual_false_accept=bool(observable and strict['actual_constraint_pass']
                                                 and strict['native_success'] is False),
                observed_actual_success_rejected=bool(observable and not strict['actual_constraint_pass']
                                                      and strict['native_success']),
                observed_command_failure=bool(observable and strict['commanded_constraint_pass']
                                              and strict['native_success'] is False),
                observed_command_success_rejected=bool(observable and not strict['commanded_constraint_pass']
                                                       and strict['native_success']))


def candidate_for(root, slot, reference):
    if slot['kind'] == 'control':
        return dict(trajectory=reference['trajectory'], source='expert', family='reference_command')
    return checked(root, dict(path=slot['candidate_path'], sha256=slot['candidate_sha256']))


def verify_observed_campaign(root):
    """Recompute committed observations, including those outside strict tracking."""
    root = Path(root)
    manifest = read(root/MANIFEST)
    if (manifest.get('kind') != KIND or manifest.get('purpose') != PURPOSE
            or manifest.get('formal_evidence_eligible') is not False
            or manifest.get('acceptance_ready') is not False or manifest.get('publication_allowed') is not False):
        raise ValueError('Observed manifests are development-only and never formal evidence')
    if manifest.get('runtime_sha256') != runtime_fingerprint():
        raise ValueError('Observed execution/classification implementation changed')
    design = read(root/'design_plan.json')
    if file_hash(root/'design_plan.json') != manifest['design_sha256']:
        raise ValueError('Observed design changed')
    reference, _, _ = validate_design(root, design)
    if (manifest.get('version') != 1 or manifest.get('split') != 'dev'
            or manifest.get('task') != design['task'] or manifest.get('scene_seed') != design['seed']
            or manifest.get('scene_sha256') != design['scene_sha256']):
        raise ValueError('Observed manifest task/scene/development role differs from design')
    if manifest['trial_plan'] != design['trial_plan']:
        raise ValueError('Observed manifest differs from its sealed trial design')
    index_path = root/'observed_trials.jsonl'
    index = [json.loads(x) for x in index_path.read_text().splitlines() if x.strip()] if index_path.exists() else []
    slots = design['trial_plan']
    if [r['trial_id'] for r in index] != [s['trial_id'] for s in slots[:len(index)]] or len(index) > len(slots):
        raise ValueError('Observed outcomes must be a unique contiguous prefix of the finite plan')
    indexed = {r['trial_id'] for r in index}
    directories = {p.name for p in (root/'trials').iterdir() if p.is_dir()} if (root/'trials').exists() else set()
    records = {p.parent.name for p in (root/'trials').glob('*/record.json')}
    if directories != indexed or records != indexed:
        raise ValueError('Uncommitted/extra observed trial evidence cannot be silently omitted from the index')
    rows, control, may_continue = [], None, True
    for slot, entry in zip(slots, index):
        if not may_continue:
            raise ValueError('Outcomes were run after a control/infrastructure stop condition')
        record = checked(root, entry)
        if (entry['path'] != f'trials/{slot["trial_id"]}/record.json'
                or record['trial_id'] != slot['trial_id']
                or record['manifest_sha256'] != file_hash(root/MANIFEST)):
            raise ValueError('Observed record belongs to another slot/manifest')
        candidate = candidate_for(root, slot, reference)
        trace, raw = checked(root, record['trajectory']), checked(root, record['outcome'])
        if (json_hash(trace['commanded']) != slot['trajectory_sha256']
                or record['trajectory_sha256'] != slot['trajectory_sha256']
                or raw['trial_id'] != slot['trial_id'] or raw['scene_sha256'] != design['scene_sha256']
                or trace.get('alignment') != ALIGNMENT or raw.get('alignment') != ALIGNMENT
                or raw.get('native_success_rule') != SUCCESS_RULE):
            raise ValueError('Observed outcome or command differs from its frozen scene/slot')
        classified = classify_observed(reference, candidate, raw, trace['actual'], control=control)
        if 'scene_trace' in record:
            scene = checked(root, record['scene_trace'])
            if scene.get('purpose') != 'diagnostic_only_not_constraint_or_training_input':
                raise ValueError('Observed scene telemetry must remain diagnostic-only')
        row = dict(classified, trial_id=slot['trial_id'], source=candidate['source'], family=candidate['family'],
                   scene_seed=design['seed'], scene_sha256=design['scene_sha256'],
                   evidence_sha256=file_hash(contained(root, entry['path'])),
                   actual_trajectory_sha256=json_hash(trace['actual']), trajectory_sha256=slot['trajectory_sha256'])
        rows.append(row)
        may_continue = bool(classified['observationally_valid'] and not raw.get('error') and not raw.get('cleanup_error'))
        if slot['kind'] == 'control':
            may_continue &= classified['control_execution_reliable']
            control = copy.deepcopy(trace['actual'])
    counts = {key: sum(bool(row[key]) for row in rows[1:]) for key in (
        'strict_eligible', 'observationally_valid', 'observed_actual_false_accept',
        'observed_actual_success_rejected', 'observed_command_failure', 'observed_command_success_rejected')}
    return dict(kind=KIND, purpose=PURPOSE, manifest_sha256=file_hash(root/MANIFEST),
                complete=len(index) == len(slots), may_continue=may_continue,
                planned=len(slots), received=len(rows), counts=counts, rows=rows,
                confidence_eligible=False, formal_evidence_eligible=False,
                acceptance_ready=False, publication_allowed=False, resume_allowed=False)
