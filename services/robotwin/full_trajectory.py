"""Fill uncovered reference motion with broad directions, never a fitted path.

Moving intervals constrain multi-step net direction. Stationary intervals
use a loose hold region. Reviewed task-specific geometric constraints take
precedence; rotations and gripper features are never invented just for coverage.
"""
import copy
import math
import re

import numpy as np

from services.robotwin.annotation import constraint_positions
from services.robotwin.constraint_coverage import POSITION, coverage_masks, coverage_report, runs

PREFIX = '__full_motion_v2__'
LEGACY_PREFIX = '__full_motion_v1__'
POLICY = {'version': 3, 'objective': 'window_direction_with_segment_region_fallback',
          'half_angle_rad': math.radians(60), 'turn_margin_rad': math.radians(5),
          'maximum_predicted_anchor_span': 32,
          'window_steps': 8, 'minimum_window_steps': 2, 'minimum_reference_progress_m': .005,
          'stationary_step_m': 1e-5, 'direction_slack_m': 1e-4, 'hold_margin_m': .02}
# Search-budget priors, not empirically established feasible or safe limits.
# Reference segmentation, activation and fallback selection are not tunable.
COVERAGE_POLICY_BOUNDS = {'half_angle_rad': (math.radians(5), math.radians(85)),
                          'hold_margin_m': (.001, .1), 'direction_slack_m': (0., .05)}
PENALTY_SCALES_M = {'stationary': .02, 'direction': .005}
MOTION_EXEMPT_TYPES = POSITION | {'axis_displacement_interval'}


def validate_motion_coverage(value, *, task, rules=None):
    """Explicit composition policy, separate from numerical tolerance tuning."""
    if (not isinstance(value, dict) or set(value) != {'version', 'task', 'mode', 'exemptions', 'rationale'}
            or type(value['version']) is not int or value['version'] != 1
            or not isinstance(task, str) or not task or value['task'] != task
            or value['mode'] not in ('independent', 'legacy_position_override')
            or not isinstance(value['rationale'], str) or not value['rationale'].strip()
            or not isinstance(value['exemptions'], list) or len(value['exemptions']) > 24):
        raise ValueError('Invalid motion_coverage policy or task context')
    if value['mode'] == 'legacy_position_override' and value['exemptions']:
        raise ValueError('Legacy motion_coverage has no selective exemptions')
    seen = set()
    for e in value['exemptions']:
        if (not isinstance(e, dict) or set(e) != {'rule_id', 'primitive', 'rationale', 'provenance'}
                or not isinstance(e['rule_id'], str) or not e['rule_id']
                or e['rule_id'].startswith((PREFIX, LEGACY_PREFIX)) or e['rule_id'] in seen
                or not isinstance(e['primitive'], str) or e['primitive'] not in MOTION_EXEMPT_TYPES
                or not isinstance(e['rationale'], str) or not e['rationale'].strip()):
            raise ValueError('motion_coverage exemptions require unique explicit template rule identities and reasons')
        source = e['provenance']
        if (not isinstance(source, dict) or set(source) != {'source', 'reference', 'sha256'}
                or source['source'] not in ('template_migration', 'manual_review')
                or not isinstance(source['reference'], str) or not source['reference'].strip()
                or not isinstance(source['sha256'], str) or re.fullmatch('[0-9a-f]{64}', source['sha256']) is None):
            raise ValueError('motion_coverage exemption provenance is missing or invalid')
        seen.add(e['rule_id'])
    if rules is not None:
        by_id = {r['id']:r for r in rules}
        for e in value['exemptions']:
            if e['rule_id'] not in by_id or by_id[e['rule_id']]['type'] != e['primitive']:
                raise ValueError('motion_coverage exemption must bind an existing position/motion rule identity and primitive in this task')
    return value


def motion_exemptions(constraints, policy, length):
    """Resolve stable template IDs to exact compiled IDs, never frame allowlists."""
    selected, audit = [], []
    for entry in policy['exemptions']:
        pattern = re.compile(re.escape(entry['rule_id'])+r'_(left|right)_([0-9]+)_([0-9]+)')
        matches = []
        for c in constraints:
            match = pattern.fullmatch(c['id'])
            if match is None:
                continue
            lo, hi = map(int, match.group(2,3))
            a = c['activation']
            expected_lo = a.get('event', {}).get('frame', a['start'])
            # Displacement can retain a prior anchor before the selected frame;
            # other exemptions must encode their exact reference selection.
            if (c['id'] != f'{entry["rule_id"]}_{match[1]}_{lo}_{hi}'
                    or c['subject'] != match[1]+'_eef' or c['type'] != entry['primitive']
                    or c['evidence']['status'] != 'accepted' or hi != a['end']-1
                    or (c['type'] != 'axis_displacement_interval' and lo != expected_lo)
                    or (c['type'] == 'axis_displacement_interval' and not a['start'] <= lo <= hi)):
                raise ValueError('motion_coverage compiled rule identity/subject/activation is inconsistent')
            matches.append(c)
        if not matches:
            raise ValueError('motion_coverage exemption has no compiled reference selection: '+entry['rule_id'])
        selected.extend(matches)
        masks = coverage_masks(matches, length)
        audit.append(dict(entry, compiled_constraints=[dict(id=c['id'], subject=c['subject'],
                          type=c['type'], activation=copy.deepcopy(c['activation']),
                          parameters=copy.deepcopy(c['parameters'])) for c in matches],
                          effective_reference_spans={arm:list(runs(masks[arm]['position_or_motion'], value=True))
                                                     for arm in ('left','right')}))
    return selected, audit


def validate_coverage_policy(override):
    """Validate only data-valued physical tolerances; never code or masking."""
    if (not isinstance(override, dict) or not override
            or set(override)-set(COVERAGE_POLICY_BOUNDS)):
        raise ValueError('coverage_policy requires a nonempty subset of allowed physical tolerance keys')
    for key, value in override.items():
        low, high = COVERAGE_POLICY_BOUNDS[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'coverage_policy {key} must be finite and in [{low}, {high}]')
    return override


def resolve_coverage_policy(override=None):
    policy = copy.deepcopy(POLICY)
    if override is not None:
        policy.update(validate_coverage_policy(override))
    return policy


def complete_constraints(handle, constraints, recording_id, *, policy=None, motion_coverage=None, task=None):
    override = copy.deepcopy(policy)
    policy = resolve_coverage_policy(override)
    result = [copy.deepcopy(c) for c in constraints if not c['id'].startswith((PREFIX, LEGACY_PREFIX))]
    length = len(handle['endpose/left_endpose'])
    if length < 2: raise ValueError('Full motion coverage needs at least two frames')
    for c in result:
        if c['type'] not in ('axis_displacement_interval', 'angular_displacement_interval'):
            continue
        a = c['activation']
        span = a.get('event', {}).get('frame', a['end']-1)-a['start']
        if (span <= POLICY['maximum_predicted_anchor_span'] or c['parameters'].get('anchor') == 'reference_frame'
                or c['evidence']['status'] != 'accepted'):
            continue
        arm = c['subject'].split('_')[0]
        pose = np.asarray(handle[f'endpose/{arm}_endpose'][a['start']], dtype=float)
        c['parameters']['anchor'] = 'reference_frame'
        if c['type'] == 'axis_displacement_interval':
            c['reference_frame']['origin_m'] = constraint_positions(pose, c['parameters']).tolist()
        else:
            c['reference_frame']['quaternion_wxyz'] = (pose[3:]/np.linalg.norm(pose[3:])).tolist()
        c['evidence']['rationale'] += (
            ' Chunk-local anchor revision: this label now uses the fixed REFERENCE pose at frame '
            f'{a["start"]}, stored in reference_frame, because its {span}-step span exceeds the 32-action chunk. '
            'This overrides the preceding predicted-start masking rule: it constrains displacement from that '
            'measured reference pose, not from an unobserved generated pose. Bounds and evaluation event are unchanged.')
    composition = copy.deepcopy(motion_coverage)
    if composition is not None:
        validate_motion_coverage(composition, task=task)
    exemptions = []
    coverage_inputs = result
    if composition is not None and composition['mode'] == 'independent':
        coverage_inputs, exemptions = motion_exemptions(result, composition, length)
    masks = coverage_masks(coverage_inputs, length)
    for arm in ('left', 'right'):
        positions = np.asarray(handle[f'endpose/{arm}_endpose'], dtype=float)[:, :3]
        if positions.shape != (length, 3) or not np.isfinite(positions).all():
            raise ValueError('Invalid physical position trajectory')
        steps = np.diff(positions, axis=0, prepend=positions[:1]); norms = np.linalg.norm(steps, axis=-1)
        for lo, hi in runs(masks[arm]['position_or_motion']):
            start = lo
            while start < hi:
                stationary = norms[start] <= POLICY['stationary_step_m']
                axis = None if stationary else steps[start]/norms[start]
                end = start+1
                while end < hi:
                    if (norms[end] <= POLICY['stationary_step_m']) != stationary: break
                    if not stationary and np.dot(steps[end]/norms[end], axis) < math.cos(POLICY['half_angle_rad']-POLICY['turn_margin_rad']): break
                    end += 1
                frame = {'origin_m': [0., 0., 0.], 'quaternion_wxyz': [1., 0., 0., 0.]}
                if stationary:
                    center = positions[start-1:end].mean(axis=0)
                    radius = float(np.linalg.norm(positions[start:end]-center, axis=-1).max())+policy['hold_margin_m']
                    frame['origin_m'] = center.tolist(); kind = 'point_distance_interval'
                    parameters = {'bounds': [0., radius]}; scale = PENALTY_SCALES_M['stationary']
                    rationale = 'Stationary reference interval: loose hold region, not per-frame position targets.'
                else:
                    kind = 'window_direction_cone'
                    # One region for this whole segment, not a moving per-frame
                    # imitation target. Used only when the chunk cannot expose
                    # a sufficiently long, nontrivial reference displacement.
                    center = positions[start-1:end].mean(axis=0)
                    radius = float(np.linalg.norm(positions[start-1:end]-center, axis=-1).max())+policy['hold_margin_m']
                    frame['origin_m'] = center.tolist()
                    parameters = {'axis': axis.tolist(), 'half_angle_rad': policy['half_angle_rad'],
                                  'bounds': [0., policy['direction_slack_m']],
                                  'window_steps': POLICY['window_steps'],
                                  'minimum_window_steps': POLICY['minimum_window_steps'],
                                  'minimum_reference_progress_m': POLICY['minimum_reference_progress_m'],
                                  'fallback_radius_m': radius}
                    scale = PENALTY_SCALES_M['direction']
                    rationale = ('Motion policy v3: broad 60-degree cone on up-to-8-step net EEF displacement, '
                                 'truncated at reference turn/stop segments and the chunk start. Windows under two '
                                 'steps or under 5 mm reference net movement use this entire segment\'s fixed '
                                 'bounding sphere plus a 20 mm prior. It is not a per-frame position target. '
                                 'All switches depend only on reference data. Zero predicted movement is allowed '
                                 'by the direction branch; progress and task success are separate. Neither the '
                                 'cone nor fallback region is physically calibrated or a safety guarantee.')
                    if policy != POLICY:
                        rationale = (
                            f'Motion policy v3 with explicit template tolerances: {math.degrees(policy["half_angle_rad"]):g}-degree '
                            f'cone and {policy["direction_slack_m"]:g} m residual slack on up-to-8-step net EEF displacement. '
                            'Reference turn/stop segmentation and short-window fallback selection remain fixed at the '
                            f'default policy. Fallback is the fixed reference-segment sphere plus {policy["hold_margin_m"]:g} m. '
                            'Zero predicted movement is allowed by the direction branch; success/progress are separate. '
                            'With nonzero residual slack, a wider angle is not necessarily more permissive for tiny '
                            'backward movements. Tolerances are not calibrated physical feasibility or safety limits.')
                result.append({'id': f'{PREFIX}{arm}_{start}_{end}', 'type': kind, 'subject': arm+'_eef',
                    'reference_frame': frame, 'parameters': parameters,
                    'activation': {'mode': 'interval', 'start': start, 'end': end},
                    'penalty': {'kind': 'squared_hinge', 'scale': scale, 'weight': 1.},
                    'evidence': {'source': 'demonstration_analysis', 'status': 'accepted', 'report_id': recording_id,
                                 'rationale': rationale+' Explicit operator prior, not a physically calibrated feasible set.'}})
                start = end
    report = coverage_report(result, length)
    if not report['complete']: raise ValueError('Full motion coverage failed')
    report['policy'] = policy
    if composition is not None:
        guards = [c for c in result if c['id'].startswith(PREFIX)]
        requirement = coverage_masks([*coverage_inputs, *guards], length)
        report['motion_composition'] = dict(policy=composition, resolved_exemptions=exemptions,
            meaning='Reference ordinary-motion/hold supervision plus declared replacement regions. Not contact, progress or task success.',
            legacy_position_coverage_report_is_separate=True,
            generated_constraints=[dict(id=c['id'], type=c['type'], subject=c['subject'], activation=c['activation']) for c in guards],
            replacement_reference_spans={arm:list(runs(masks[arm]['position_or_motion'], value=True)) for arm in ('left','right')},
            generated_reference_spans={arm:list(runs(coverage_masks(guards,length)[arm]['position_or_motion'], value=True)) for arm in ('left','right')},
            complete=all(all(requirement[a]['position_or_motion'][1:]) for a in ('left','right')))
        if not report['motion_composition']['complete']:
            raise ValueError('Explicit motion composition leaves uncovered generated frames')
    if override is not None:
        report['coverage_override'] = override
        report['reference_partition_policy'] = {key: POLICY[key] for key in (
            'half_angle_rad', 'turn_margin_rad', 'stationary_step_m')}
        report['penalty_scales_m'] = dict(PENALTY_SCALES_M)
    return result, report
