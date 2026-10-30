"""Deterministic motion phases from reference EEF positions only.

The selected phase is not an inferred contact event. No generated actions,
constraint coverage masks, scene state, or native success results are inputs.
"""
import copy
import hashlib
import json
import math

import numpy as np


POLICY = dict(version=1, position_source='reference_eef_origin_world_xyz',
              stationary_step_m=1e-5, turn_from_first_step_rad=math.radians(55),
              minimum_steps=2, minimum_eef_net_displacement_m=.0001,
              minimum_derived_axis_displacement_m=.0001,
              maximum_predicted_anchor_span=32)


class InvalidReferenceMotionPhase(ValueError):
    def __init__(self, reason, audit):
        super().__init__(reason)
        self.audit = copy.deepcopy(audit)
        self.audit.update(valid=False, invalid_reason=reason)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def select_reference_motion_phase(positions, after_frame):
    """Select the last qualifying full-reference segment without clipping it."""
    p = np.asarray(positions, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all()
            or type(after_frame) is not int or not 0 <= after_frame < len(p)):
        raise ValueError('Reference motion selection requires finite XYZ and a valid fixed reference boundary')
    steps = np.diff(p, axis=0); norms = np.linalg.norm(steps, axis=-1)
    if not np.isfinite(steps).all() or not np.isfinite(norms).all():
        raise ValueError('Non-finite reference displacement')
    audit = dict(policy=copy.deepcopy(POLICY), reference_positions_sha256=json_hash(p.tolist()),
                 after_frame=after_frame, after_semantics='segment anchor >= reference boundary; no segment clipping',
                 generated_actions_used=False, native_outcome_used=False, contact_phase_proven=False,
                 segments=[])
    start = 1
    while start < len(p):
        stationary = bool(norms[start-1] <= POLICY['stationary_step_m'])
        direction = None if stationary else steps[start-1]/norms[start-1]
        end = start+1
        while end < len(p):
            if bool(norms[end-1] <= POLICY['stationary_step_m']) != stationary:
                break
            if not stationary and np.dot(steps[end-1]/norms[end-1], direction) < math.cos(POLICY['turn_from_first_step_rad']):
                break
            end += 1
        delta = p[end-1]-p[start-1]; distance = float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or not math.isfinite(distance):
            raise InvalidReferenceMotionPhase('Non-finite reference segment net displacement', audit)
        segment = dict(first_action=start, end_exclusive=end, anchor=start-1,
                       steps=end-start, stationary=stationary,
                       eef_net_displacement_m=delta.tolist(), eef_net_distance_m=distance)
        segment['qualifies'] = (not stationary and segment['anchor'] >= after_frame
                               and segment['steps'] >= POLICY['minimum_steps']
                               and distance >= POLICY['minimum_eef_net_displacement_m'])
        audit['segments'].append(segment)
        start = end
    candidates = [s for s in audit['segments'] if s['qualifies']]
    if not candidates:
        raise InvalidReferenceMotionPhase('No qualifying reference motion segment after the fixed boundary', audit)
    selected = copy.deepcopy(candidates[-1])
    audit.update(selected=selected, valid=True, occurrence='last')
    return selected, audit


def reference_segment_axis(positions, segment, audit):
    """World direction of the selected physical point's reference net motion."""
    points = np.asarray(positions, dtype=float)
    delta = points[segment['end_exclusive']-1]-points[segment['anchor']]
    distance = float(np.linalg.norm(delta))
    if not np.isfinite(delta).all() or not math.isfinite(distance) or distance < POLICY['minimum_derived_axis_displacement_m']:
        raise InvalidReferenceMotionPhase('Selected reference point has near-zero or invalid net displacement; no alternate phase is selected', audit)
    return (delta/distance).tolist(), distance
