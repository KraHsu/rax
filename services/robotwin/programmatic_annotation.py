"""Data-only task templates instantiated against each reference trajectory.

No model, simulator, generated actions, arbitrary Python, or physical-success
inference is involved. Margins are the existing explicit reference priors.
"""
import copy
import hashlib
import json
import math
import re

import numpy as np

from services.robotwin.annotation import constraint_positions, reference_values, rotation, validate_reference_bounds
from services.robotwin.constraint_schema import KINDS, LOCAL_POINT_KINDS, SCHEMA_VERSION, keys, read_embedded, validate_bundle, validate_local_point, vector
from services.robotwin.full_trajectory import PREFIX, LEGACY_PREFIX, MOTION_EXEMPT_TYPES, complete_constraints, validate_coverage_policy, validate_motion_coverage

POLICY = {'position_margin_m': .005, 'angle_margin_rad': .0872664626, 'gripper_margin': .02}
SELECTORS = ('events', 'gripper_plateaus', 'terminal', 'position_extremum', 'joint_gripper_plateaus')
GRIPPER_PLATEAU_EPSILON = 1e-6


def margin_unit(kind):
    if kind in ('gripper_interval', 'gripper_pair_max_interval'):
        return 'fraction', 1.
    if kind in ('axis_angle_interval', 'angular_displacement_interval'):
        return 'rad', math.pi
    return 'm', 1.


def allowed_margin(rule, default):
    """Physical freedom is separate from the penalty's numerical scale."""
    value = rule.get('allowed_margin')
    if value is None:
        return default, default
    keys(value, ('lower', 'upper', 'unit'))
    unit, maximum = margin_unit(rule['type'])
    if value['unit'] != unit or any(type(value[k]) not in (int, float)
            or not math.isfinite(value[k]) or not 0 <= value[k] <= maximum for k in ('lower', 'upper')):
        raise ValueError('Invalid physical allowed_margin or unit')
    return float(value['lower']), float(value['upper'])


class OutsideTemplate(ValueError):
    """A new reference pattern needs RAX review, not an invented annotation."""


def template_hash(template):
    return hashlib.sha256(json.dumps(template, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def legacy_motion_coverage(template):
    """Declare stable old rule identities before adding independent geometry."""
    validate_template(template)
    if template.get('motion_coverage', {}).get('mode', 'legacy_position_override') != 'legacy_position_override':
        raise ValueError('Automatic motion migration requires a legacy parent template')
    digest = template_hash(template)
    return dict(version=1, task=template['task'], mode='independent',
        rationale='Explicit migration preserves the legacy reference position-rule replacement masks; later geometry is conjoined with ordinary motion.',
        exemptions=[dict(rule_id=r['id'], primitive=r['type'], rationale='This original rule replaced ordinary motion under the legacy position-coverage policy; freeze its identity and primitive, not episode frame numbers.',
                         provenance=dict(source='template_migration', reference=f'programmatic_template:{template["task"]}:{digest}', sha256=digest))
                    for r in template['rules'] if r['type'] in MOTION_EXEMPT_TYPES])


def validate_reference_roles(declarations):
    """Template-side roles; no role token survives into the action-loss schema."""
    if not isinstance(declarations, list) or not 1 <= len(declarations) <= 2:
        raise ValueError('reference_roles requires one or two explicit declarations')
    names = set()
    for declaration in declarations:
        keys(declaration, ('role', 'source', 'event_signature', 'rationale'))
        name = declaration['role']
        if (not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', name)
                or name in names):
            raise ValueError('Reference role names must be unique stable identifiers')
        names.add(name)
        if declaration['source'] != 'reference_event_signature':
            raise ValueError('Role source must be a fixed reference event signature')
        signature = declaration['event_signature']
        events = signature.split(',') if isinstance(signature, str) else []
        if (not 1 <= len(events) <= 32 or any(e not in ('close', 'open') for e in events)
                or any(a == b for a, b in zip(events, events[1:]))):
            raise ValueError('Role event signature must be a nonempty alternating close/open sequence')
        if not isinstance(declaration['rationale'], str) or not declaration['rationale'].strip():
            raise ValueError('Explain the native/reference basis for a role')
    return names


def resolve_reference_roles(template, bundle, arms, meta):
    """Bind once from the successful recording; ambiguity is a review blocker.

    The hash covers the decoded reference arrays, action contract and recording
    identity. It is not the HDF5 byte hash (which changes upon annotation).
    Generated trajectories are never an input to this function.
    """
    declarations = template.get('reference_roles')
    if declarations is None:
        return {}, None
    observed = {arm: ','.join('close' if e['direction'] == 'decreasing' else 'open'
                             for e in data['events']) for arm, data in arms.items()}
    bindings = {}
    for declaration in declarations:
        matches = [arm for arm, data in arms.items() if data['active']
                   and observed[arm] == declaration['event_signature']]
        if len(matches) != 1:
            raise OutsideTemplate(f'Reference role {declaration["role"]} needs exactly one active arm; found {matches}')
        arm = matches[0]
        if arm in bindings.values():
            raise OutsideTemplate('Multiple reference roles bind the same arm')
        bindings[declaration['role']] = arm
    source = dict(action_contract=bundle['action_contract'],
                  recording_id=bundle['provenance']['recording_id'], expert_provenance=meta,
                  arms={arm: dict(pose=data['pose'].tolist(), gripper=data['grip'].tolist())
                        for arm, data in arms.items()})
    audit = dict(version=1, source='fixed_successful_reference',
                 declarations=copy.deepcopy(declarations), declarations_sha256=template_hash(declarations),
                 reference_sha256=template_hash(source),
                 reference_hash_format='canonical_json_decoded_pose_gripper_contract_recording_expert_v1',
                 recording_id=source['recording_id'], expert_provenance=copy.deepcopy(meta),
                 observed_arms={arm: dict(active=data['active'], event_signature=observed[arm],
                                         events=copy.deepcopy(data['events'])) for arm, data in arms.items()},
                 bindings=bindings, generated_actions_used=False)
    return bindings, audit


def validate_template(t):
    keys(t, ('version', 'task', 'rationale', 'signatures', 'rules'), ('coverage_policy', 'reference_roles', 'motion_coverage'))
    role_names = validate_reference_roles(t['reference_roles']) if 'reference_roles' in t else set()
    if 'coverage_policy' in t:
        validate_coverage_policy(t['coverage_policy'])
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
             ('start_anchor', 'required', 'allowed_margin', 'reference_phase'))
        if not isinstance(r['id'], str) or not r['id'] or r['id'] in ids or r['id'].startswith((PREFIX, LEGACY_PREFIX)):
            raise ValueError('Rule IDs must be unique nonempty strings')
        ids.add(r['id'])
        subjects = {'active_arms', 'left', 'right', 'both_grippers'} | {'role:'+name for name in role_names}
        if (r['type'] not in KINDS or r['type'] in ('motion_direction_cone', 'window_direction_cone')
                or not isinstance(r['subjects'], str) or r['subjects'] not in subjects):
            raise ValueError('Unsupported primitive or subject')
        if (r['subjects'] == 'both_grippers') != (r['type'] == 'gripper_pair_max_interval'):
            raise ValueError('Pair max requires exactly both_grippers')
        allowed_margin(r, 0.)
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
        keys(r['parameters'], params, ('local_point',) if r['type'] in LOCAL_POINT_KINDS else ())
        for key in params:
            vector(r['parameters'][key], 3, unit=True)
        if 'local_point' in r['parameters']:
            validate_local_point(r['parameters']['local_point'], r['type'])
        s = r['selector']; kind = s.get('kind') if isinstance(s, dict) else None
        if kind not in SELECTORS:
            raise ValueError('Unsupported selector; absolute frame numbers are forbidden')
        fields = {'kind'}
        if kind == 'events': fields.update(('direction', 'occurrence'))
        if kind in ('gripper_plateaus', 'joint_gripper_plateaus'): fields.update(('state', 'after'))
        if kind == 'position_extremum': fields.update(('axis', 'which', 'after'))
        keys(s, fields)
        if kind == 'events' and (s['direction'] not in ('increasing', 'decreasing', 'any') or s['occurrence'] not in ('all', 'first', 'last')):
            raise ValueError('Invalid event selector')
        if kind == 'gripper_plateaus' and s['state'] not in ('closed', 'open', 'all'):
            raise ValueError('Invalid plateau state')
        if 'after' in s and s['after'] not in ('episode_start', 'first_close', 'last_close'):
            raise ValueError('Invalid selection anchor')
        if kind == 'joint_gripper_plateaus' or r['type'] == 'gripper_pair_max_interval':
            if (r['type'] != 'gripper_pair_max_interval' or s != {'kind': 'joint_gripper_plateaus', 'state': 'before_joint_closed', 'after': 'episode_start'}
                    or r['origin'] != 'world' or r['orientation'] != 'world' or not r.get('required', True)):
                raise ValueError('Pair max uses fixed required reference-before-joint-closed plateaus and identity frame')
        if 'reference_phase' in r:
            if (r['reference_phase'] != 'after_joint_closed' or r['type'] != 'gripper_interval'
                    or s != {'kind': 'gripper_plateaus', 'state': 'all', 'after': 'episode_start'}):
                raise ValueError('reference_phase only clips existing all-gripper plateaus after joint closure')
        if kind == 'position_extremum':
            vector(s['axis'], 3, unit=True)
            if s['which'] not in ('min', 'max'): raise ValueError('Invalid extremum')
        if kind == 'gripper_plateaus' and r.get('start_anchor', 'selection') != 'selection':
            raise ValueError('An interval displacement must start at its selected interval')
        if (r['type'] in ('axis_displacement_interval', 'angular_displacement_interval')
                and kind != 'gripper_plateaus' and r.get('start_anchor', 'selection') == 'selection'):
            raise ValueError('Point displacement needs an explicit earlier start_anchor')
    if 'motion_coverage' in t:
        validate_motion_coverage(t['motion_coverage'], task=t['task'], rules=t['rules'])
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
    if selector['state'] == 'all':
        # Freeze maximal stable reference runs, including partial openings.
        # Compare to each run's first value, so many small changes cannot drift
        # through an arbitrarily broad "plateau". Single frames are excluded.
        runs = []; lo = start
        for index in range(start+1, length+1):
            if index == length or abs(arm['grip'][index]-arm['grip'][lo]) > GRIPPER_PLATEAU_EPSILON:
                if index-lo >= 2:
                    runs.append({'mode': 'interval', 'start': lo, 'end': index})
                lo = index
        return runs
    mask = np.abs(arm['grip']-(0 if selector['state'] == 'closed' else 1)) <= GRIPPER_PLATEAU_EPSILON
    mask[:start] = False
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [{'mode': 'interval', 'start': int(lo), 'end': int(hi)}
            for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if hi-lo >= 2]



def joint_gripper_windows(arms):
    """Reference-only max plateaus before the first stable common full closure.

    Max can stay constant while the other hand ramps; that freedom is intended.
    This reference boundary is not a claim about physical contact or grasp time.
    """
    grips = np.stack([arms[a]['grip'] for a in ('left', 'right')], -1)
    closed = (np.abs(grips) <= GRIPPER_PLATEAU_EPSILON).all(-1)
    starts = np.flatnonzero(closed[:-1] & closed[1:])
    if not len(starts):
        raise OutsideTemplate('Missing stable reference joint-closed boundary')
    boundary = int(starts[0]); values = grips.max(-1)
    windows = []; lo = 0
    for index in range(1, boundary+1):
        if index == boundary or abs(values[index]-values[lo]) > GRIPPER_PLATEAU_EPSILON:
            if index-lo >= 2:
                windows.append(dict(mode='interval', start=lo, end=index))
            lo = index
    return windows, dict(first_joint_closed_stable=boundary, epsilon=GRIPPER_PLATEAU_EPSILON,
                        minimum_stable_frames=2, plateau_feature='max(left_open_fraction,right_open_fraction)',
                        activation_source='reference_only', windows=copy.deepcopy(windows))


def instantiate(handle, template):
    """Return validated constraints and an audit; never copy another episode's bounds."""
    validate_template(template)
    bundle, arms, signature = trajectory_data(handle)
    meta = json.loads(handle.attrs['robotwin_expert_provenance'])
    if meta['task'] != template['task']: raise OutsideTemplate('Task mismatch')
    if signature not in template['signatures']: raise OutsideTemplate(f'Unreviewed event signature: {signature}')
    role_bindings, role_audit = resolve_reference_roles(template, bundle, arms, meta)
    constraints = []; thash = template_hash(template)
    uses_joint = any(r['type'] == 'gripper_pair_max_interval' or 'reference_phase' in r for r in template['rules'])
    joint_windows, joint_audit = joint_gripper_windows(arms) if uses_joint else ([], None)
    removed = []; retained = []
    for rule in template['rules']:
        subjects = [a for a in arms if arms[a]['active']] if rule['subjects'] == 'active_arms' else [rule['subjects']]
        if rule['subjects'].startswith('role:'):
            subjects = [role_bindings[rule['subjects'][5:]]]
        selected_count = 0
        for subject in subjects:
            pair = rule['type'] == 'gripper_pair_max_interval'
            arm = arms['left'] if pair else arms[subject]
            try: windows = copy.deepcopy(joint_windows) if pair else selections(rule['selector'], arm)
            except OutsideTemplate:
                if rule.get('required', True): raise
                windows = []
            original_windows = {}
            if 'reference_phase' in rule:
                boundary = joint_audit['first_joint_closed_stable']
                clipped = []
                for w in windows:
                    removed.extend((rule['id'], subject, i) for i in range(max(1, w['start']), min(w['end'], boundary)))
                    if w['end'] > boundary:
                        v = dict(w, start=max(w['start'], boundary)); clipped.append(v)
                        original_windows[(v['start'], v['end'])] = w
                        retained.append(dict(rule_id=rule['id'], subject=subject, activation=v))
                windows = clipped
            for window in windows:
                selected_count += 1
                # A reference event's frame, not the preceding threshold sample.
                bounds_window = original_windows.get((window['start'], window['end']), window)
                lo = bounds_window.get('event', {}).get('frame', bounds_window['start']); hi = bounds_window['end']-1
                positions = constraint_positions(arm['pose'], rule['parameters'])
                origin = [0., 0., 0.] if rule['origin'] == 'world' else positions[lo if rule['origin'] == 'selection_start' else hi].tolist()
                quat = [1., 0., 0., 0.] if rule['orientation'] == 'world' else arm['pose'][lo, 3:].tolist()
                quat = (np.asarray(quat)/np.linalg.norm(quat)).tolist()
                activation = copy.deepcopy(window)
                if rule.get('start_anchor', 'selection') != 'selection':
                    activation['start'] = anchor(rule['start_anchor'], arm)
                    if activation['start'] > lo: raise OutsideTemplate('Displacement anchor is after selection')
                    if activation['start'] == hi == lo: raise OutsideTemplate('Zero-length displacement has no training signal')
                kind = rule['type']
                scale = POLICY['gripper_margin'] if kind in ('gripper_interval', 'gripper_pair_max_interval') else (
                    POLICY['angle_margin_rad'] if kind in ('axis_angle_interval', 'angular_displacement_interval') else POLICY['position_margin_m'])
                c = {'id': f'{rule["id"]}_{subject}_{lo}_{hi}', 'type': kind, 'subject': subject if pair else subject+'_eef',
                     'reference_frame': {'origin_m': origin, 'quaternion_wxyz': quat},
                     'parameters': dict(copy.deepcopy(rule['parameters']), bounds=[0., 0.]), 'activation': activation,
                     'penalty': {'kind': 'squared_hinge', 'scale': scale, 'weight': 1.},
                     'evidence': {'source': 'demonstration_analysis', 'status': 'accepted',
                         'report_id': bundle['provenance']['recording_id'],
                         'rationale': f'Programmatic template {template["task"]} sha256={thash}; rule={rule["id"]}. '
                                      + rule['rationale'] + ' Bounds are recomputed from this reference with explicit operator margins; '
                                      'not a physically calibrated feasible region or proof of contact/grasp/support. '
                                      'Displacement requires its original start anchor in the training chunk.'}}
                if rule['subjects'].startswith('role:'):
                    c['evidence']['rationale'] += (f' Fixed reference role {rule["subjects"][5:]} -> {subject}; '
                        f'reference_sha256={role_audit["reference_sha256"]}; '
                        f'declarations_sha256={role_audit["declarations_sha256"]}.')
                # Preserve the original platform's numerical set when clipping
                # its activation, including tiny within-epsilon reference drift.
                original_activation = c['activation']
                c['activation'] = bounds_window if original_windows else original_activation
                values = reference_values(handle, c)
                c['activation'] = original_activation
                lower_margin, upper_margin = allowed_margin(rule, scale)
                lower, upper = float(values.min()-lower_margin), float(values.max()+upper_margin)
                if kind in ('point_distance_interval', 'radial_distance_interval', 'gripper_interval', 'gripper_pair_max_interval', 'axis_angle_interval', 'angular_displacement_interval'):
                    lower = max(0., lower)
                if kind in ('gripper_interval', 'gripper_pair_max_interval'): upper = min(1., upper)
                if kind in ('axis_angle_interval', 'angular_displacement_interval'): upper = min(math.pi, upper)
                c['parameters']['bounds'] = [lower, upper]
                constraints.append(c)
        if selected_count == 0 and rule.get('required', True): raise OutsideTemplate(f'Missing selection for {rule["id"]}')
    if removed:
        covered = {i for c in constraints if c['type'] == 'gripper_pair_max_interval' and c['parameters']['bounds'][0] > 0
                   for i in range(max(1, c['activation']['start']), c['activation']['end'])}
        missing = sorted({i for _, _, i in removed}-covered)
        if missing:
            raise OutsideTemplate(f'Joint replacement leaves removed reference frames uncovered: {missing}')
    if joint_audit is not None:
        joint_audit.update(replaced_pre_frames=[dict(rule_id=r, subject=a, frame=i) for r,a,i in removed], retained_post_windows=retained)
    if not constraints or len(constraints) > 128 or not any(c['type'] not in ('gripper_interval', 'gripper_pair_max_interval') for c in constraints):
        raise OutsideTemplate('Need 1..128 constraints with at least one geometric feature')
    constraints, coverage = complete_constraints(handle, constraints, bundle['provenance']['recording_id'],
                                                policy=template.get('coverage_policy'), motion_coverage=template.get('motion_coverage'),
                                                task=template['task'])
    generated = dict(bundle, constraints=constraints, schema_version=SCHEMA_VERSION)
    validate_bundle(generated, len(arms['left']['grip']))
    audit = validate_reference_bounds(generated, handle)
    return constraints, {'template_sha256': thash, 'signature': signature, 'policy': POLICY,
                         'reference_audit': audit, 'coverage': coverage,
                         **({'reference_roles': role_audit} if role_audit is not None else {}),
                         **({'joint_gripper_reference': joint_audit} if uses_joint else {})}


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
