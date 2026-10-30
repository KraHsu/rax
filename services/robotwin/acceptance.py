"""CPU evidence gate for counterfactual constraint acceptance.

Native rollouts supply outcomes; this module independently evaluates measured
trajectories against *reference-selected* constraints. No perturbation name is
treated as a failure label. This is a small, explicit pilot gate, not proof of a
continuous feasible set or a high-confidence task-success guarantee.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from pathlib import Path

import numpy as np

from services.robotwin.annotation import reference_values, rotation
from services.robotwin.constraint_coverage import coverage_report
from services.robotwin.constraint_schema import validate_bundle

ACCEPTANCE_VERSION = 2
ALIGNMENT = 'reference_command_index_v1'
SUCCESS_RULE = 'native_ever_success'
FAMILIES = ('position', 'orientation', 'gripper_timing', 'coupled')
DEFAULT_PROTOCOL = {
    'version': 'counterfactual_pilot_v1',
    'minimum_scenes': 3,
    'baseline_repeats_per_seed': 2,
    'minimum_success_variants': 3,
    'minimum_task_failures': 6,
    'minimum_failure_families': 2,
    'maximum_false_accept_rate': 0.0,
    'maximum_false_reject_rate': 0.2,
    'violation_tolerance': 1e-6,
    'required_families': list(FAMILIES),
}

# Recomputed arithmetic can differ by a few ulps across Python/NumPy builds.
# These tolerances are only for evidence consistency, never physical limits.
DERIVED_RTOL = 1e-10
DERIVED_ATOL = 1e-12


def derived_values_equal(actual, expected):
    """Compare derived JSON values without demanding floating-point bit identity.

    Artifact hashes remain exact. Keys, sequence lengths, booleans, strings and
    discrete integers remain exact; only finite real arithmetic receives the
    fixed numerical tolerance. This does not relax pass/fail decisions.
    """
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (isinstance(actual, dict) and isinstance(expected, dict)
                and actual.keys() == expected.keys()
                and all(derived_values_equal(actual[k], expected[k]) for k in actual))
    if isinstance(actual, list) or isinstance(expected, list):
        return (isinstance(actual, list) and isinstance(expected, list)
                and len(actual) == len(expected)
                and all(derived_values_equal(a, b) for a, b in zip(actual, expected)))
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual is expected
    if isinstance(actual, numbers.Integral) or isinstance(expected, numbers.Integral):
        return (isinstance(actual, numbers.Integral) and isinstance(expected, numbers.Integral)
                and actual == expected)
    if isinstance(actual, numbers.Real) or isinstance(expected, numbers.Real):
        return (isinstance(actual, numbers.Real) and isinstance(expected, numbers.Real)
                and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual, expected, rel_tol=DERIVED_RTOL, abs_tol=DERIVED_ATOL))
    return type(actual) is type(expected) and actual == expected


def json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def annotation_policy_fingerprint():
    from services.robotwin.constraint_schema import SCHEMA_VERSION
    from services.robotwin.full_trajectory import POLICY as motion_policy
    from services.robotwin.programmatic_annotation import POLICY as numeric_policy

    # Includes numeric behavior, not only a manually maintained version label.
    source_names = ('annotation.py', 'constraint_schema.py', 'constraint_coverage.py', 'acceptance_replay.py',
                    'full_trajectory.py', 'programmatic_annotation.py')
    return json_hash(dict(schema=SCHEMA_VERSION, numeric=numeric_policy, motion=motion_policy,
                          source_sha256={name: file_hash(Path(__file__).with_name(name)) for name in source_names}))


def evaluator_fingerprint():
    return file_hash(__file__)


def validate_protocol(protocol=None):
    value = copy.deepcopy(DEFAULT_PROTOCOL if protocol is None else protocol)
    if set(value) != set(DEFAULT_PROTOCOL) or value['version'] != DEFAULT_PROTOCOL['version']:
        raise ValueError('Explicit complete counterfactual pilot protocol required')
    for name in ('minimum_scenes', 'baseline_repeats_per_seed', 'minimum_success_variants',
                 'minimum_task_failures', 'minimum_failure_families'):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError(f'Invalid positive protocol count: {name}')
    for name in ('maximum_false_accept_rate', 'maximum_false_reject_rate'):
        if isinstance(value[name], bool) or not 0 <= value[name] < 1:
            raise ValueError(f'Invalid protocol rate: {name}')
    if not np.isfinite(value['violation_tolerance']) or not 0 <= value['violation_tolerance'] <= 1e-4:
        raise ValueError('violation_tolerance is numerical slack, not an adjustable physical margin')
    if (not value['required_families'] or len(set(value['required_families'])) != len(value['required_families'])
            or set(value['required_families']) - set(FAMILIES)):
        raise ValueError('Invalid required perturbation families')
    return value


def validate_acceptance_bundle(bundle, length):
    """Check the training-compatible geometry without labeling validation as train.

    The existing schema deliberately rejects non-training episodes. Only this
    private structural-validation copy changes split; the stored bundle stays
    validation and cannot be consumed by read_embedded as a training episode.
    """
    if (bundle.get('provenance', {}).get('split') != 'validation'
            or bundle['provenance'].get('variant') != 'clean_heldout'
            or bundle['provenance'].get('source') != 'acceptance_reference_not_for_training'):
        raise ValueError('Acceptance bundle must be an explicit validation-only reference')
    structural = copy.deepcopy(bundle)
    structural['provenance']['split'] = 'train'
    structural['provenance']['variant'] = 'clean_50'
    validate_bundle(structural, length)
    if bundle['action_contract']['quaternion_order'] != 'wxyz':
        raise ValueError('Measured acceptance poses must use WXYZ')
    accepted = [c for c in bundle['constraints'] if c['evidence']['status'] == 'accepted']
    if not accepted or not coverage_report(accepted, length)['complete']:
        raise ValueError('Acceptance reference lacks complete temporal motion coverage')
    return accepted


def _trajectory(value):
    arrays, length = {}, None
    if set(value) != {'left', 'right'}:
        raise ValueError('Measured trajectory must contain both arms')
    for arm in ('left', 'right'):
        pose = np.asarray(value[arm]['pose'], dtype=np.float64)
        grip = np.asarray(value[arm]['gripper'], dtype=np.float64)
        if pose.ndim != 2 or pose.shape[1] != 7 or grip.shape != (len(pose),) or len(pose) < 2:
            raise ValueError('Expected pose[N,7] and gripper[N], including initial state')
        if not np.isfinite(pose).all() or not np.isfinite(grip).all() or np.any((grip < 0) | (grip > 1)):
            raise ValueError('Nonfinite pose or invalid measured gripper opening')
        rotation(pose[:, 3:])
        if length is not None and length != len(pose):
            raise ValueError('Arm trajectories are not aligned')
        length = len(pose)
        arrays[f'endpose/{arm}_endpose'] = pose
        arrays[f'endpose/{arm}_gripper'] = grip
    return arrays, length


def trajectory_violations(bundle, reference, actual, *, tolerance=1e-6):
    """Score complete command-index-aligned traces, with fixed reference labels.

    EEF poses are measured; RoboTwin gripper fields are nominal command opening
    fractions. They are not finger force or proof of contact/grasp stability.
    """
    ref, length = _trajectory(reference)
    measured, actual_length = _trajectory(actual)
    if actual_length != length:
        raise ValueError('Incomplete measured trajectory; no truncation or DTW acceptance')
    constraints = validate_acceptance_bundle(bundle, length)
    results = []
    for c in constraints:
        a, p = c['activation'], c['parameters']
        indices = np.arange(a['start'], a['end'])
        if a['mode'] == 'endpoint':
            indices = indices[-1:]
        elif a['mode'] == 'reference_event':
            event = a['event']
            index = event['frame']
            arm = c['subject'].split('_')[0]
            grip = ref[f'endpose/{arm}_gripper'][index-1:index+1]
            sign = 1 if event['direction'] == 'increasing' else -1
            signed = sign * (grip-event['threshold'])
            if len(grip) != 2 or not (signed[0] < 0 <= signed[1]):
                raise ValueError('Annotated event does not occur in the frozen reference')
            indices = np.array([index])
        active = indices > 0  # The measured initial state is not a generated action.
        if not active.any():
            continue
        if c['type'] == 'window_direction_cone':
            arm = c['subject'].split('_')[0]
            rp, ap = ref[f'endpose/{arm}_endpose'], measured[f'endpose/{arm}_endpose']
            anchors = np.maximum(indices-p['window_steps'], a['start']-1)
            usable = ((indices-anchors >= p['minimum_window_steps'])
                      & (np.linalg.norm(rp[indices, :3]-rp[anchors, :3], axis=-1)
                         >= p['minimum_reference_progress_m']))
            step = ap[indices, :3]-ap[anchors, :3]
            axis = rotation(c['reference_frame']['quaternion_wxyz']) @ np.asarray(p['axis'])
            axial = step @ axis
            lateral = np.linalg.norm(step-axial[:, None]*axis, axis=-1)
            cone = np.maximum.reduce((lateral-np.tan(p['half_angle_rad'])*axial, -axial, np.zeros_like(axial)))
            distance = np.linalg.norm(ap[indices, :3]-c['reference_frame']['origin_m'], axis=-1)
            fallback = np.maximum(distance-p['fallback_radius_m'], 0)+p['bounds'][1]
            values = np.where(usable, cone, fallback)
        else:
            values = reference_values(measured, c)
        values = values[active]
        low, high = p['bounds']
        violations = np.maximum(low-values, 0)+np.maximum(values-high, 0)
        if not np.isfinite(violations).all():
            raise ValueError('Nonfinite constraint violation')
        results.append(dict(id=c['id'], type=c['type'], samples=len(values),
                            max_violation=float(violations.max()),
                            rejected_samples=int(np.count_nonzero(violations > tolerance))))
    if not results:
        raise ValueError('No constraints activated on generated action frames')
    maximum = max(r['max_violation'] for r in results)
    return dict(passed=maximum <= tolerance, max_violation=maximum, constraints=results)


def _artifact(root, record, name):
    path = (Path(root)/record[name+'_path']).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError(f'Invalid {name} artifact path')
    if file_hash(path) != record[name+'_sha256']:
        raise ValueError(f'Changed {name} artifact')
    return json.loads(path.read_text())


def validate_manifest(manifest):
    from services.robotwin.programmatic_annotation import template_hash, validate_template

    for key in ('task', 'template_sha256', 'annotation_policy_sha256', 'native_commit'):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f'Missing acceptance context: {key}')
    if (manifest.get('version') != 1 or manifest.get('mode') != 'demo_clean'
            or manifest.get('split') != 'validation' or manifest.get('alignment') != ALIGNMENT
            or manifest.get('native_success_rule') != SUCCESS_RULE):
        raise ValueError('Acceptance requires explicit held-out clean aligned native evidence')
    if manifest['annotation_policy_sha256'] != annotation_policy_fingerprint():
        raise ValueError('Acceptance evidence uses another annotation implementation/policy')
    validate_template(manifest['template'])
    if template_hash(manifest['template']) != manifest['template_sha256'] or manifest['template']['task'] != manifest['task']:
        raise ValueError('Sealed template does not match its identity')
    for name in ('excluded_training_seeds', 'excluded_training_scene_sha256'):
        if not isinstance(manifest.get(name), list) or not manifest[name]:
            raise ValueError(f'Explicit training exclusion inventory required: {name}')
    if any(type(seed) is not int for seed in manifest['excluded_training_seeds']):
        raise ValueError('Training exclusion seeds must be integers')
    if any(not isinstance(scene, str) or not scene for scene in manifest['excluded_training_scene_sha256']):
        raise ValueError('Training exclusion scene IDs must be nonempty strings')
    plan = manifest.get('trial_plan')
    if not isinstance(plan, list) or not plan:
        raise ValueError('A sealed complete trial plan is required')
    ids = set()
    for item in plan:
        if not isinstance(item.get('trial_id'), str) or not item['trial_id'] or item['trial_id'] in ids:
            raise ValueError('Trial IDs must be unique and nonempty')
        if type(item.get('seed')) is not int or item['seed'] in manifest['excluded_training_seeds']:
            raise ValueError('Trial seed is invalid or overlaps training')
        if item.get('variant_family') not in ('baseline', *FAMILIES):
            raise ValueError('Unknown trial family')
        validate_case(item.get('case'), item['variant_family'])
        ids.add(item['trial_id'])
    if 'reference_artifacts' in manifest:
        references = manifest['reference_artifacts']
        if (len({r['seed'] for r in references}) != len(references)
                or {r['seed'] for r in references} != {p['seed'] for p in plan}):
            raise ValueError('Frozen references must cover each planned seed exactly once')
    return manifest


def validate_case(case, family):
    allowed = {'id', 'variant_family', 'position_axis', 'position_offset_m', 'position_window',
               'rotation_axis', 'rotation_rad', 'gripper_shift', 'gripper_force'}
    if (not isinstance(case, dict) or not {'id', 'variant_family'} <= set(case)
            or set(case)-allowed or case['variant_family'] != family
            or not isinstance(case['id'], str) or not case['id']):
        raise ValueError('Every trial requires its complete frozen intervention case')
    categories = set()
    for prefix, quantity, category in [('position', 'position_offset_m', 'position'),
                                        ('rotation', 'rotation_rad', 'orientation')]:
        if quantity in case or prefix+'_axis' in case:
            if (quantity not in case or type(case.get(prefix+'_axis')) is not int
                    or case[prefix+'_axis'] not in (0, 1, 2)
                    or type(case[quantity]) not in (int, float)
                    or not np.isfinite(case[quantity]) or case[quantity] == 0):
                raise ValueError('Invalid frozen geometric intervention')
            categories.add(category)
    if 'position_window' in case and ('position' not in categories or case['position_window'] != 'closed_phase'):
        raise ValueError('Unsupported intervention window')
    if 'gripper_shift' in case:
        if type(case['gripper_shift']) is not int or case['gripper_shift'] == 0:
            raise ValueError('Gripper timing shift must be a nonzero integer')
        categories.add('gripper_timing')
    if 'gripper_force' in case:
        if type(case['gripper_force']) not in (int, float) or not 0 <= case['gripper_force'] <= 1:
            raise ValueError('Gripper intervention must be a nominal opening fraction')
        categories.add('gripper_timing')
    if ((family == 'baseline' and categories) or (family == 'coupled' and len(categories) < 2)
            or (family not in ('baseline', 'coupled') and categories != {family})):
        raise ValueError('Intervention parameters do not substantiate their declared family')
    return case


def evaluate_trial(record, evidence_root, manifest, *, protocol=None):
    protocol = validate_protocol(protocol)
    if record.get('version') != 1:
        raise ValueError('Unsupported trial record version')
    for key in ('task', 'template_sha256', 'annotation_policy_sha256', 'native_commit', 'mode', 'split',
                'alignment', 'native_success_rule'):
        if record.get(key) != manifest[key]:
            raise ValueError(f'Trial context mismatch: {key}')
    planned = next((p for p in manifest['trial_plan'] if p['trial_id'] == record.get('trial_id')), None)
    if planned is None or any(record.get(k) != planned[k] for k in ('seed', 'variant_family')):
        raise ValueError('Trial is outside its sealed plan')
    if 'reference_artifacts' in manifest:
        frozen = next(r for r in manifest['reference_artifacts'] if r['seed'] == record['seed'])
        if record.get('reference_path') != frozen['path'] or record.get('reference_sha256') != frozen['sha256']:
            raise ValueError('Trial reference differs from the preselected frozen artifact')
    if record.get('scene_sha256') in manifest['excluded_training_scene_sha256']:
        raise ValueError('Held-out scene overlaps training')
    if not isinstance(record.get('scene_sha256'), str) or not record['scene_sha256']:
        raise ValueError('Missing initial scene identity')
    raw = _artifact(evidence_root, record, 'outcome')
    for key in ('trial_id', 'task', 'seed', 'scene_sha256', 'variant_family'):
        if raw.get(key) != record[key]:
            raise ValueError(f'Native outcome identity mismatch: {key}')
    for key in ('planner_success', 'realized', 'execution_complete'):
        if type(raw.get(key)) is not bool:
            raise ValueError(f'Native outcome needs boolean {key}')
    result = {k: record[k] for k in ('trial_id', 'seed', 'scene_sha256', 'variant_family')}
    if not raw['execution_complete']:
        return dict(result, eligible=False, reason='incomplete_execution',
                    native_success=raw.get('native_success'), constraint_pass=None)
    if type(raw.get('native_success')) is not bool or raw.get('ever_success') is not raw['native_success']:
        raise ValueError('Native success must be the independently recorded ever-success predicate')
    reference = _artifact(evidence_root, record, 'reference')
    trajectory = _artifact(evidence_root, record, 'trajectory')
    provenance = reference['provenance']
    for key in ('task', 'seed', 'scene_sha256', 'mode', 'split'):
        if provenance.get(key) != record[key]:
            raise ValueError(f'Reference identity mismatch: {key}')
    if json_hash(reference['bundle']) != record.get('bundle_sha256'):
        raise ValueError('Reference bundle hash mismatch')
    _verify_instantiation(reference, manifest['template'])
    tracking = _verify_execution(record, evidence_root, manifest, planned, raw, reference, trajectory, protocol)
    if not tracking['eligible']:
        return dict(result, eligible=False, reason=tracking['reason'],
                    native_success=raw['native_success'], constraint_pass=None)
    baseline = trajectory_violations(reference['bundle'], reference['trajectory'], reference['trajectory'],
                                     tolerance=protocol['violation_tolerance'])
    if not baseline['passed']:
        raise ValueError('Frozen constraints reject the successful source reference')
    score = trajectory_violations(reference['bundle'], reference['trajectory'], trajectory['actual'],
                                  tolerance=protocol['violation_tolerance'])
    return dict(result, eligible=True, native_success=raw['native_success'], constraint_pass=score['passed'],
                max_violation=score['max_violation'], constraint_details=score['constraints'],
                reference_sha256=record['reference_sha256'], bundle_sha256=record['bundle_sha256'])


def _verify_execution(record, evidence_root, manifest, planned, raw, reference, trajectory, protocol):
    """Recompute execution, including commands and effect relative to baseline."""
    from services.robotwin.acceptance_replay import arrays, intervene, tracking_report

    expected = intervene(reference['trajectory'], planned['case'])
    if not derived_values_equal(expected, trajectory['commanded']):
        raise ValueError('Commanded trace differs from the sealed intervention case')
    actual, ref = arrays(trajectory['actual']), arrays(reference['trajectory'])
    length = len(ref['left']['pose'])
    if len(actual['left']['pose']) != length:
        raise ValueError('Complete outcome has an incomplete actual trace')
    for arm in ('left', 'right'):
        ap, rp = actual[arm]['pose'][0], ref[arm]['pose'][0]
        if (not np.allclose(ap[:3], rp[:3], rtol=0, atol=1e-5)
                or abs(float(ap[3:] @ rp[3:])) < 1-1e-6
                or abs(actual[arm]['gripper'][0]-ref[arm]['gripper'][0]) > 1e-5):
            raise ValueError('Initial measured robot state differs from reference')
    rows = raw['planner_results']
    expected_pairs = {(arm, frame) for arm in ('left', 'right') for frame in range(1, length)}
    planning_ok = (len(rows) == len(expected_pairs)
                   and {(r['arm'], r['frame']) for r in rows} == expected_pairs
                   and all(r['status'] == 'Success' for r in rows))
    if raw['planner_success'] is not planning_ok:
        raise ValueError('Planner summary differs from recorded native planner results')
    first = next((p for p in manifest['trial_plan']
                  if p['seed'] == record['seed'] and p['variant_family'] == 'baseline'), None)
    if first is None:
        raise ValueError('Each seed needs a preregistered baseline')
    baseline = None
    if first['trial_id'] != record['trial_id']:
        records = [json.loads(line) for line in (Path(evidence_root)/'trials.jsonl').read_text().splitlines() if line.strip()]
        baseline_records = [r for r in records if r['trial_id'] == first['trial_id']]
        if len(baseline_records) != 1:
            raise ValueError('First baseline is missing or duplicated')
        base_record = baseline_records[0]
        checked = evaluate_trial(base_record, evidence_root, manifest, protocol=protocol)
        if not (checked['eligible'] and checked['native_success'] and checked['constraint_pass']):
            raise ValueError('First baseline is not a valid reference for intervention tracking')
        if (base_record['scene_sha256'] != record['scene_sha256']
                or base_record['reference_sha256'] != record['reference_sha256']):
            raise ValueError('Intervention tracking baseline belongs to another scene/reference')
        baseline = _artifact(evidence_root, base_record, 'trajectory')['actual']
    track = tracking_report(trajectory['actual'], expected, baseline=baseline,
                            reference=reference['trajectory'] if baseline is not None else None)
    if raw['realized'] is not track['realized'] or not derived_values_equal(raw.get('tracking'), track):
        raise ValueError('Tracking/realization summary differs from independently measured effect')
    if planned['variant_family'] != 'baseline' and json_hash(expected) == json_hash(reference['trajectory']):
        return dict(eligible=False, reason='intervention_had_no_command_effect')
    return dict(eligible=planning_ok and track['realized'], reason='planner_failure_or_unrealized_intervention')


def _verify_instantiation(reference, template):
    """Do not accept caller-provided loose labels under an unrelated template hash."""
    from services.robotwin.constraint_schema import HDF5_KEY
    from services.robotwin.programmatic_annotation import instantiate

    class Handle(dict):
        pass

    arrays, length = _trajectory(reference['trajectory'])
    validate_acceptance_bundle(reference['bundle'], length)
    source = copy.deepcopy(reference['bundle'])
    source['provenance'].update(split='train', variant='clean_50')
    handle = Handle(arrays)
    handle[HDF5_KEY] = np.asarray(json.dumps(source))
    handle.attrs = {'robotwin_expert_provenance': json.dumps(reference['provenance'])}
    constraints, _ = instantiate(handle, template)
    if not derived_values_equal(constraints, reference['bundle']['constraints']):
        raise ValueError('Reference labels do not match deterministic template/policy instantiation')


def assess_evidence(evidence_root, *, protocol=None, expected_task=None, expected_template_sha256=None):
    root = Path(evidence_root)
    protocol = validate_protocol(protocol)
    manifest_path = root/'experiment_manifest.json'
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    if expected_task is not None and manifest['task'] != expected_task:
        raise ValueError('Acceptance task mismatch')
    if expected_template_sha256 is not None and manifest['template_sha256'] != expected_template_sha256:
        raise ValueError('Acceptance template mismatch')
    records_path = root/'trials.jsonl'
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()] if records_path.exists() else []
    ids = [r.get('trial_id') for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate trial outcomes are forbidden; retries need distinct planned IDs')
    planned_ids = {p['trial_id'] for p in manifest['trial_plan']}
    if set(ids)-planned_ids:
        raise ValueError('Unplanned evidence cannot be cherry-picked into the gate')
    results, errors = [], []
    for record in records:
        try:
            results.append(evaluate_trial(record, root, manifest, protocol=protocol))
        except (ValueError, KeyError, FileNotFoundError, TypeError) as exc:
            errors.append(dict(trial_id=record.get('trial_id'), error=f'{type(exc).__name__}: {exc}'))
    valid = [r for r in results if r['eligible']]
    baselines = [r for r in valid if r['variant_family'] == 'baseline']
    positives = [r for r in valid if r['variant_family'] != 'baseline' and r['native_success']]
    negatives = [r for r in valid if r['variant_family'] != 'baseline' and not r['native_success']]
    fa = sum(r['constraint_pass'] for r in negatives)
    fr = sum(not r['constraint_pass'] for r in positives)
    accepted_successes = len(positives)-fr
    reasons = []
    if planned_ids-set(ids):
        reasons.append('missing_planned_trials')
    if errors:
        reasons.append('invalid_evidence')
    if any(r['variant_family'] == 'baseline' and not r['eligible'] for r in results):
        reasons.append('baseline_execution_invalid')
    seeds = {p['seed'] for p in manifest['trial_plan']}
    scenes = {r['scene_sha256'] for r in valid}
    if len(seeds) < protocol['minimum_scenes'] or len(scenes) < protocol['minimum_scenes']:
        reasons.append('insufficient_distinct_heldout_seeds_or_scenes')
    # A seed must identify one immutable scene and one immutable frozen reference.
    for seed in seeds:
        subset = [r for r in valid if r['seed'] == seed]
        if (len({r['scene_sha256'] for r in subset}) != 1
                or len({r['reference_sha256'] for r in subset}) != 1
                or len({r['bundle_sha256'] for r in subset}) != 1):
            reasons.append(f'inconsistent_reference_or_scene_seed_{seed}')
        base = [r for r in baselines if r['seed'] == seed]
        if len(base) < protocol['baseline_repeats_per_seed'] or any(not r['native_success'] or not r['constraint_pass'] for r in base):
            reasons.append(f'baseline_not_reproducible_seed_{seed}')
    if len(positives) < protocol['minimum_success_variants']:
        reasons.append('insufficient_successful_variations')
    if len(negatives) < protocol['minimum_task_failures']:
        reasons.append('insufficient_native_task_failures')
    if len({r['seed'] for r in positives}) < protocol['minimum_scenes']:
        reasons.append('success_variations_do_not_cover_heldout_seeds')
    if len({r['seed'] for r in negatives}) < protocol['minimum_scenes']:
        reasons.append('task_failures_do_not_cover_heldout_seeds')
    observed_families = {r['variant_family'] for r in valid}
    if set(protocol['required_families'])-observed_families:
        reasons.append('missing_perturbation_families')
    if len({r['variant_family'] for r in negatives}) < protocol['minimum_failure_families']:
        reasons.append('insufficient_failure_family_diversity')
    far = fa/len(negatives) if negatives else None
    frr = fr/len(positives) if positives else None
    if far is not None and far > protocol['maximum_false_accept_rate']:
        reasons.append('failed_variations_pass_constraints')
    if frr is not None and frr > protocol['maximum_false_reject_rate']:
        reasons.append('successful_variations_rejected')
    return dict(version=ACCEPTANCE_VERSION, task=manifest['task'], template_sha256=manifest['template_sha256'],
                annotation_policy_sha256=manifest['annotation_policy_sha256'], evaluator_sha256=evaluator_fingerprint(),
                protocol=protocol, protocol_sha256=json_hash(protocol), manifest_sha256=file_hash(manifest_path),
                records_sha256=file_hash(records_path) if records_path.exists() else None,
                ready=not reasons, status='ready' if not reasons else 'blocked', reasons=sorted(set(reasons)),
                counts=dict(planned=len(planned_ids), received=len(records), valid=len(valid), baselines=len(baselines),
                            successful_variations=len(positives), native_task_failures=len(negatives),
                            excluded_trials=len(results)-len(valid), invalid_evidence=len(errors),
                            false_accepts=fa, false_rejects=fr, accepted_successes=accepted_successes),
                false_accept_rate=far, false_reject_rate=frr, errors=errors, trials=results,
                success_among_accepted_variations=(accepted_successes/(accepted_successes+fa)
                                                   if accepted_successes+fa else None),
                meaning='Finite selected held-out clean pilot cases, not an IID population estimate, '
                        'continuous feasible-set proof, or success guarantee.')
