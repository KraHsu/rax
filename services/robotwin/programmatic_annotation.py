"""Data-only task templates instantiated against each reference trajectory.

No model, simulator, generated actions, arbitrary Python, or physical-success
inference is involved. Margins are the existing explicit reference priors.
"""
import copy
import hashlib
import json
import math

import numpy as np

from services.robotwin.annotation import reference_values, rotation, validate_reference_bounds
from services.robotwin.constraint_schema import KINDS, keys, read_embedded, validate_bundle, vector

POLICY = {'position_margin_m': .005, 'angle_margin_rad': .0872664626, 'gripper_margin': .02}
SELECTORS = ('events', 'gripper_plateaus', 'terminal', 'position_extremum')


class OutsideTemplate(ValueError):
    """A new reference pattern needs RAX review, not an invented annotation."""


def template_hash(template):
    return hashlib.sha256(json.dumps(template, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_template(t):
    keys(t, ('version', 'task', 'rationale', 'signatures', 'rules'))
    if (type(t['version']) is not int or t['version'] != 1 or not isinstance(t['task'], str) or not t['task']
            or not isinstance(t['rationale'], str) or not t['rationale']):
        raise ValueError('Invalid task template identity')
    if not isinstance(t['signatures'], list) or not t['signatures']:
        raise ValueError('Explicit observed event signatures required')
    for signature in t['signatures']:
        if not isinstance(signature, list) or not 1 <= len(signature) <= 2:
            raise ValueError('Signature lists each active arm, with arm names abstracted')
        if any(not isinstance(s, str) or (s and any(e not in ('close', 'open') for e in s.split(','))) for s in signature):
            raise ValueError('Invalid event signature')
        if signature != sorted(signature):
            raise ValueError('Sort signatures to support left/right variants')
    if not isinstance(t['rules'], list) or not 1 <= len(t['rules']) <= 24:
        raise ValueError('Expected 1..24 task-relevant rules')
    ids = set()
    for r in t['rules']:
        keys(r, ('id', 'type', 'subjects', 'selector', 'origin', 'orientation', 'parameters', 'rationale'),
             ('start_anchor', 'required'))
        if not isinstance(r['id'], str) or not r['id'] or r['id'] in ids:
            raise ValueError('Rule IDs must be unique nonempty strings')
        ids.add(r['id'])
        if r['type'] not in KINDS or r['subjects'] not in ('active_arms', 'left', 'right'):
            raise ValueError('Unsupported primitive or subject')
        if not isinstance(r['rationale'], str) or not r['rationale']:
            raise ValueError('Explain the selected feature')
        if 'required' in r and type(r['required']) is not bool:
            raise ValueError('required must be boolean')
        if r['origin'] not in ('world', 'selection_start', 'selection_end') or r['orientation'] not in ('world', 'selection_start'):
            raise ValueError('Use a measured reference frame or world frame')
        if r.get('start_anchor', 'selection') not in ('selection', 'first_close', 'last_close', 'episode_start'):
            raise ValueError('Invalid displacement anchor')
        if 'start_anchor' in r and r['type'] not in ('axis_displacement_interval', 'angular_displacement_interval'):
            raise ValueError('start_anchor is only for displacement primitives')
        params = set()
        if r['type'] in ('axis_position_interval', 'radial_distance_interval', 'axis_displacement_interval'):
            params.add('axis')
        if r['type'] == 'axis_angle_interval':
            params.update(('local_axis', 'reference_axis'))
        keys(r['parameters'], params)
        for value in r['parameters'].values():
            vector(value, 3, unit=True)
        s = r['selector']; kind = s.get('kind') if isinstance(s, dict) else None
        if kind not in SELECTORS:
            raise ValueError('Unsupported selector; absolute frame numbers are forbidden')
        fields = {'kind'}
        if kind == 'events': fields.update(('direction', 'occurrence'))
        if kind == 'gripper_plateaus': fields.update(('state', 'after'))
        if kind == 'position_extremum': fields.update(('axis', 'which', 'after'))
        keys(s, fields)
        if kind == 'events' and (s['direction'] not in ('increasing', 'decreasing', 'any') or s['occurrence'] not in ('all', 'first', 'last')):
            raise ValueError('Invalid event selector')
        if kind == 'gripper_plateaus' and s['state'] not in ('closed', 'open'):
            raise ValueError('Invalid plateau state')
        if 'after' in s and s['after'] not in ('episode_start', 'first_close', 'last_close'):
            raise ValueError('Invalid selection anchor')
        if kind == 'position_extremum':
            vector(s['axis'], 3, unit=True)
            if s['which'] not in ('min', 'max'): raise ValueError('Invalid extremum')
        if kind == 'gripper_plateaus' and r.get('start_anchor', 'selection') != 'selection':
            raise ValueError('An interval displacement must start at its selected interval')
        if (r['type'] in ('axis_displacement_interval', 'angular_displacement_interval')
                and kind != 'gripper_plateaus' and r.get('start_anchor', 'selection') == 'selection'):
            raise ValueError('Point displacement needs an explicit earlier start_anchor')
    return t


def trajectory_data(handle):
    bundle = read_embedded(handle)
    if bundle is None or bundle['provenance'].get('native_success') is not True:
        raise ValueError('Require successful clean reference trajectory')
    if bundle['action_contract']['quaternion_order'] != 'wxyz':
        raise ValueError('This numeric adapter requires WXYZ')
    length = len(handle['endpose/left_endpose'])
    if length < 2: raise ValueError('Reference trajectory too short')
    arms = {}
    for arm in ('left', 'right'):
        pose = np.asarray(handle[f'endpose/{arm}_endpose'], dtype=float)
        grip = np.asarray(handle[f'endpose/{arm}_gripper'], dtype=float).reshape(-1)
        if pose.shape != (length, 7) or grip.shape != (length,) or not np.isfinite(pose).all() or not np.isfinite(grip).all():
            raise ValueError('Invalid reference arrays')
        if np.any(grip < 0) or np.any(grip > 1): raise ValueError('Gripper fraction outside [0,1]')
        rot = rotation(pose[:, 3:])
        events = []
        for direction, sign in (('decreasing', -1), ('increasing', 1)):
            events += [{'frame': int(i), 'direction': direction, 'threshold': .5}
                       for i in np.flatnonzero((sign*(grip[:-1]-.5) < 0) & (sign*(grip[1:]-.5) >= 0))+1]
        events.sort(key=lambda e: e['frame'])
        relative = rot[0].T @ rot
        angle = np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1))
        active = (np.linalg.norm(pose[:, :3]-pose[0, :3], axis=-1).max() > .01
                  or angle.max() > math.radians(5) or np.ptp(grip) > .05)
        arms[arm] = {'pose': pose, 'grip': grip, 'events': events, 'active': bool(active)}
    signature = sorted(','.join('close' if e['direction'] == 'decreasing' else 'open' for e in a['events'])
                       for a in arms.values() if a['active'])
    return bundle, arms, signature


def anchor(name, arm):
    if name == 'episode_start': return 0
    closes = [e['frame'] for e in arm['events'] if e['direction'] == 'decreasing']
    if not closes: raise OutsideTemplate('Missing reference closure anchor')
    return closes[0] if name == 'first_close' else closes[-1]


def selections(selector, arm):
    kind = selector['kind']; length = len(arm['grip'])
    if kind == 'terminal': return [{'mode': 'endpoint', 'start': length-1, 'end': length}]
    if kind == 'events':
        events = [e for e in arm['events'] if selector['direction'] in ('any', e['direction'])]
        if selector['occurrence'] == 'first': events = events[:1]
        if selector['occurrence'] == 'last': events = events[-1:]
        return [{'mode': 'reference_event', 'start': e['frame']-1, 'end': e['frame']+1, 'event': dict(e)} for e in events]
    start = anchor(selector['after'], arm)
    if kind == 'position_extremum':
        values = arm['pose'][start:, :3] @ np.asarray(selector['axis'])
        index = start + int(np.argmin(values) if selector['which'] == 'min' else np.argmax(values))
        return [{'mode': 'endpoint', 'start': index, 'end': index+1}]
    mask = np.abs(arm['grip']-(0 if selector['state'] == 'closed' else 1)) <= 1e-6
    mask[:start] = False
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [{'mode': 'interval', 'start': int(lo), 'end': int(hi)}
            for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if hi-lo >= 2]


def instantiate(handle, template):
    """Return validated constraints and an audit; never copy another episode's bounds."""
    validate_template(template)
    bundle, arms, signature = trajectory_data(handle)
    meta = json.loads(handle.attrs['robotwin_expert_provenance'])
    if meta['task'] != template['task']: raise OutsideTemplate('Task mismatch')
    if signature not in template['signatures']: raise OutsideTemplate(f'Unreviewed event signature: {signature}')
    constraints = []; thash = template_hash(template)
    for rule in template['rules']:
        subjects = [a for a in arms if arms[a]['active']] if rule['subjects'] == 'active_arms' else [rule['subjects']]
        selected_count = 0
        for subject in subjects:
            arm = arms[subject]
            try: windows = selections(rule['selector'], arm)
            except OutsideTemplate:
                if rule.get('required', True): raise
                windows = []
            for window in windows:
                selected_count += 1
                # A reference event's frame, not the preceding threshold sample.
                lo = window.get('event', {}).get('frame', window['start']); hi = window['end']-1
                origin = [0., 0., 0.] if rule['origin'] == 'world' else arm['pose'][lo if rule['origin'] == 'selection_start' else hi, :3].tolist()
                quat = [1., 0., 0., 0.] if rule['orientation'] == 'world' else arm['pose'][lo, 3:].tolist()
                quat = (np.asarray(quat)/np.linalg.norm(quat)).tolist()
                activation = copy.deepcopy(window)
                if rule.get('start_anchor', 'selection') != 'selection':
                    activation['start'] = anchor(rule['start_anchor'], arm)
                    if activation['start'] > lo: raise OutsideTemplate('Displacement anchor is after selection')
                    if activation['start'] == hi == lo: raise OutsideTemplate('Zero-length displacement has no training signal')
                kind = rule['type']
                scale = POLICY['gripper_margin'] if kind == 'gripper_interval' else (
                    POLICY['angle_margin_rad'] if kind in ('axis_angle_interval', 'angular_displacement_interval') else POLICY['position_margin_m'])
                c = {'id': f'{rule["id"]}_{subject}_{lo}_{hi}', 'type': kind, 'subject': subject+'_eef',
                     'reference_frame': {'origin_m': origin, 'quaternion_wxyz': quat},
                     'parameters': dict(rule['parameters'], bounds=[0., 0.]), 'activation': activation,
                     'penalty': {'kind': 'squared_hinge', 'scale': scale, 'weight': 1.},
                     'evidence': {'source': 'demonstration_analysis', 'status': 'accepted',
                         'report_id': bundle['provenance']['recording_id'],
                         'rationale': f'Programmatic template {template["task"]} sha256={thash}; rule={rule["id"]}. '
                                      + rule['rationale'] + ' Bounds are recomputed from this reference with explicit operator margins; '
                                      'not a physically calibrated feasible region or proof of contact/grasp/support. '
                                      'Displacement requires its original start anchor in the training chunk.'}}
                values = reference_values(handle, c)
                lower, upper = float(values.min()-scale), float(values.max()+scale)
                if kind in ('point_distance_interval', 'radial_distance_interval', 'gripper_interval', 'axis_angle_interval', 'angular_displacement_interval'):
                    lower = max(0., lower)
                if kind == 'gripper_interval': upper = min(1., upper)
                if kind in ('axis_angle_interval', 'angular_displacement_interval'): upper = min(math.pi, upper)
                c['parameters']['bounds'] = [lower, upper]
                constraints.append(c)
        if selected_count == 0 and rule.get('required', True): raise OutsideTemplate(f'Missing selection for {rule["id"]}')
    if not constraints or len(constraints) > 128 or not any(c['type'] != 'gripper_interval' for c in constraints):
        raise OutsideTemplate('Need 1..128 constraints with at least one geometric feature')
    generated = dict(bundle, constraints=constraints)
    validate_bundle(generated, len(arms['left']['grip']))
    audit = validate_reference_bounds(generated, handle)
    return constraints, {'template_sha256': thash, 'signature': signature, 'policy': POLICY, 'reference_audit': audit}


def instantiate_review(review_path, template):
    """Identical numeric path for a small local CSV review, without image HDF5."""
    from pathlib import Path
    from services.robotwin.constraint_schema import HDF5_KEY
    root = Path(review_path); review = json.loads((root/'review.json').read_text())
    values = np.genfromtxt(root/'trajectory.csv', delimiter=',', names=True)
    class Handle(dict): pass
    h = Handle({HDF5_KEY: np.asarray(json.dumps(review['bundle_template']))})
    h.attrs = {'robotwin_expert_provenance': json.dumps(review['expert_provenance'])}
    for arm in ('left', 'right'):
        h[f'endpose/{arm}_endpose'] = np.stack([values[f'{arm}_{name}'] for name in ('x','y','z','qw','qx','qy','qz')], -1)
        h[f'endpose/{arm}_gripper'] = values[f'{arm}_gripper']
    return instantiate(h, template)
