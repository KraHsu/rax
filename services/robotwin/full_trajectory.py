"""Fill uncovered reference motion with broad directions, never a fitted path.

Moving intervals constrain only adjacent-step direction. Stationary intervals
use a loose hold region. Reviewed task-specific geometric constraints take
precedence; rotations and gripper features are never invented just for coverage.
"""
import copy
import math

import numpy as np

from services.robotwin.constraint_coverage import coverage_masks, coverage_report, runs

PREFIX = '__full_motion_v1__'
POLICY = {'version': 1, 'half_angle_rad': math.radians(60), 'turn_margin_rad': math.radians(5),
          'stationary_step_m': 1e-5, 'direction_slack_m': 1e-4, 'hold_margin_m': .02}


def complete_constraints(handle, constraints, recording_id):
    result = [copy.deepcopy(c) for c in constraints if not c['id'].startswith(PREFIX)]
    length = len(handle['endpose/left_endpose'])
    if length < 2: raise ValueError('Full motion coverage needs at least two frames')
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
                    radius = float(np.linalg.norm(positions[start:end]-center, axis=-1).max())+POLICY['hold_margin_m']
                    frame['origin_m'] = center.tolist(); kind = 'point_distance_interval'
                    parameters = {'bounds': [0., radius]}; scale = POLICY['hold_margin_m']
                    rationale = 'Stationary reference interval: loose hold region, not per-frame position targets.'
                else:
                    kind = 'motion_direction_cone'
                    parameters = {'axis': axis.tolist(), 'half_angle_rad': POLICY['half_angle_rad'],
                                  'bounds': [0., POLICY['direction_slack_m']]}
                    scale = .005
                    rationale = ('Moving reference interval: broad 60-degree half-angle cone on adjacent physical EEF '
                                 'displacements. No absolute path, speed or per-frame position targets. Zero motion is '
                                 'allowed by this direction constraint; task endpoints and the base loss still matter.')
                result.append({'id': f'{PREFIX}{arm}_{start}_{end}', 'type': kind, 'subject': arm+'_eef',
                    'reference_frame': frame, 'parameters': parameters,
                    'activation': {'mode': 'interval', 'start': start, 'end': end},
                    'penalty': {'kind': 'squared_hinge', 'scale': scale, 'weight': 1.},
                    'evidence': {'source': 'demonstration_analysis', 'status': 'accepted', 'report_id': recording_id,
                                 'rationale': rationale+' Explicit operator prior, not a physically calibrated feasible set.'}})
                start = end
    report = coverage_report(result, length)
    if not report['complete']: raise ValueError('Full motion coverage failed')
    report['policy'] = POLICY
    return result, report
