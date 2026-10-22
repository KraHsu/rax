"""Pure interventions and measured features for clean expert tolerance experiments."""
import copy
import numpy as np


def perturb_program(program, delta):
    """Freeze all expert targets; change only the declared grasp/final targets.

    Grasp XYZ shifts both approach and closure poses. The lift and placement
    targets remain frozen. Gripper opening replaces the closure target and is
    held by native control thereafter. Terminal Z changes the last move only.
    """
    allowed = {'grasp_x','grasp_y','grasp_z','opening','terminal_z'}
    if not isinstance(delta,dict) or set(delta)-allowed:
        raise ValueError('Unknown intervention; use world-metre grasp/terminal offsets and opening fraction')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) for v in delta.values()):
        raise ValueError('Intervention values must be finite numbers')
    if not 0 <= delta.get('opening',0.) <= 1:
        raise ValueError('Opening must be in [0,1], without native clipping')
    result = copy.deepcopy(program)
    shift = np.array([delta.get(k, 0.) for k in ('grasp_x', 'grasp_y', 'grasp_z')])
    for group in result[0]:
        for action in group['actions']:
            if action['action'] == 'move':
                action['target_pose'][:3] = (np.array(action['target_pose'][:3])+shift).tolist()
            elif action['target_gripper_pos'] == 0:
                action['target_gripper_pos'] = float(delta.get('opening', 0.))
    final = [a for stage in result for group in stage for a in group['actions'] if a['action'] == 'move'][-1]
    final['target_pose'][2] += float(delta.get('terminal_z', 0.))
    return result


def features(samples, arm):
    poses = np.array([s[arm+'_pose'] for s in samples])
    openings = np.array([s[arm+'_opening'] for s in samples])
    crossings = np.flatnonzero((openings[:-1] > .5) & (openings[1:] <= .5))+1
    event = int(crossings[0]) if len(crossings) else None
    grasp = [s for s in samples if s.get('stage') == 0]
    return {'sample_count': len(samples), 'closure_frame': event,
            'closure_xyz_m': poses[event, :3].tolist() if event is not None else None,
            'grasp_end_xyz_m': list(grasp[-1][arm+'_pose'][:3]) if grasp else None,
            'terminal_xyz_m': poses[-1, :3].tolist(),
            'vertical_displacement_m': float(poses[-1, 2]-poses[event, 2]) if event is not None else None,
            'terminal_opening_command': float(openings[-1])}


def realization(reference, observed, delta):
    """Separate commanded offsets from actual geometry; collisions can saturate."""
    measured, errors = {}, {}
    valid = True
    if any(k.startswith('grasp_') for k in delta):
        actual = np.array(observed['grasp_end_xyz_m'])-reference['grasp_end_xyz_m']
        for i, key in enumerate(('grasp_x','grasp_y','grasp_z')):
            target = delta.get(key,0.)
            measured[key] = float(actual[i]); errors[key] = float(actual[i]-target)
            tolerance = max(.001,abs(target)*.1) if key in delta else .002
            valid = valid and abs(errors[key]) <= tolerance
    if 'terminal_z' in delta:
        measured['terminal_z'] = observed['terminal_xyz_m'][2]-reference['terminal_xyz_m'][2]
        errors['terminal_z'] = measured['terminal_z']-delta['terminal_z']
        valid = valid and abs(errors['terminal_z']) <= max(.001,abs(delta['terminal_z'])*.1)
    return {'realized':bool(valid),'measured_offset_m':measured,'tracking_error_m':errors}


def initial_cases():
    cases = []
    for axis in ('grasp_x', 'grasp_y', 'grasp_z'):
        for mm in (-40, -20, -10, -5, 5, 10, 20, 40):
            cases.append({axis:mm/1000})
    for opening in (.02, .05, .1, .2, .4, .6, .8):
        cases.append({'opening':opening})
    for mm in (-80, -40, -20, -10, -5, 5, 10, 20, 40, 80):
        cases.append({'terminal_z':mm/1000})
    return cases


def bracket(results, axis, sign):
    """First observed non-success after zero; never bridge a failed point."""
    candidates = sorted((r for r in results
                        if set(r['delta']) == {axis} and r['delta'][axis]*sign > 0
                        and r['outcome'] in ('success', 'task_failure', 'planner_failure')),
                        key=lambda r:abs(r['delta'][axis]))
    groups = {}
    for r in candidates: groups.setdefault(abs(r['delta'][axis]),[]).append(r)
    good = 0.
    for distance, repeated in groups.items():
        passing = [r['outcome']=='success' and r.get('realized',True) for r in repeated]
        if not all(passing):
            result = next(r for r in repeated if r['outcome']!='success' or not r.get('realized',True))
            kind = ('inconsistent' if any(passing) else 'unrealized_offset' if result['outcome']=='success'
                    and not result.get('realized',True) else result['outcome'])
            return {'pass':sign*good, 'nonpass':sign*distance, 'nonpass_kind':kind}
        good = distance
    return {'pass':sign*good, 'nonpass':None, 'nonpass_kind':None}
