"""Fill uncovered reference motion with broad directions, never a fitted path.

Moving intervals constrain multi-step net direction. Stationary intervals
use a loose hold region. Reviewed task-specific geometric constraints take
precedence; rotations and gripper features are never invented just for coverage.
"""
import copy
import math

import numpy as np

from services.robotwin.constraint_coverage import coverage_masks, coverage_report, runs

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


def complete_constraints(handle, constraints, recording_id, *, policy=None):
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
            c['reference_frame']['origin_m'] = pose[:3].tolist()
        else:
            c['reference_frame']['quaternion_wxyz'] = (pose[3:]/np.linalg.norm(pose[3:])).tolist()
        c['evidence']['rationale'] += (
            ' Chunk-local anchor revision: this label now uses the fixed REFERENCE pose at frame '
            f'{a["start"]}, stored in reference_frame, because its {span}-step span exceeds the 32-action chunk. '
            'This overrides the preceding predicted-start masking rule: it constrains displacement from that '
            'measured reference pose, not from an unobserved generated pose. Bounds and evaluation event are unchanged.')
    masks = coverage_masks(result, length)
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
    if override is not None:
        report['coverage_override'] = override
        report['reference_partition_policy'] = {key: POLICY[key] for key in (
            'half_angle_rad', 'turn_margin_rad', 'stationary_step_m')}
        report['penalty_scales_m'] = dict(PENALTY_SCALES_M)
    return result, report
