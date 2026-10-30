"""Command-indexed clean-scene interventions for offline acceptance experiments.

No intervention is labelled successful or failed in advance. The simulator's
original success predicate supplies outcomes; numerical tracking is separately
required before interpreting an intervention. This module has no GPU imports.
"""
import math

import numpy as np

ALIGNMENT = 'reference_command_index_v1'
SUCCESS_RULE = 'native_ever_success'
TRACKING = {'position_m': .005, 'rotation_rad': math.radians(5), 'gripper_fraction': .05}
RELATIVE_TRACKING = {'position_m': .001, 'rotation_rad': math.radians(.5),
                     'gripper_fraction': .01, 'relative_fraction': .35}


def arrays(trajectory):
    result = {}
    length = None
    for arm in ('left', 'right'):
        pose = np.asarray(trajectory[arm]['pose'], dtype=float)
        grip = np.asarray(trajectory[arm]['gripper'], dtype=float)
        if pose.ndim != 2 or pose.shape[1] != 7 or grip.shape != (len(pose),):
            raise ValueError('Expected matching [frames, XYZ+WXYZ] and gripper arrays')
        if len(pose) < 2 or not np.isfinite(pose).all() or not np.isfinite(grip).all():
            raise ValueError('Trajectory must be finite and have at least two frames')
        if np.any((grip < 0) | (grip > 1)) or not np.allclose(np.linalg.norm(pose[:, 3:], axis=-1), 1, atol=1e-5):
            raise ValueError('Expected unit WXYZ quaternions and [0,1] nominal gripper commands')
        if length is not None and length != len(pose):
            raise ValueError('Left/right frame counts differ')
        length = len(pose)
        result[arm] = {'pose': pose.copy(), 'gripper': grip.copy()}
    return result


def plain_trajectory(trajectory):
    return {arm: {key: np.asarray(value).tolist() for key, value in row.items()}
            for arm, row in trajectory.items()}


def quaternion_multiply(left, right):
    left, right = np.broadcast_arrays(np.asarray(left), np.asarray(right))
    w, x, y, z = np.moveaxis(left, -1, 0)
    a, b, c, d = np.moveaxis(right, -1, 0)
    return np.stack((w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                     w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a), -1)


def quaternion_inverse(value):
    result = np.asarray(value).copy()
    result[..., 1:] *= -1
    return result


def quaternion_distance(left, right):
    # acos(dot) magnifies a one-ulp dot error into ~1e-8 rad near identity.
    # atan2 preserves small rotations; the ratio also cancels quaternion scale.
    # abs(w) identifies q and -q and returns the shortest SO(3) angle [0, pi].
    relative = quaternion_multiply(np.asarray(left), quaternion_inverse(right))
    return 2*np.arctan2(np.linalg.norm(relative[..., 1:], axis=-1), np.abs(relative[..., 0]))


def active_arms(reference):
    reference = arrays(reference)
    active = []
    for arm, row in reference.items():
        pose, grip = row['pose'], row['gripper']
        if (np.linalg.norm(pose[:, :3]-pose[0, :3], axis=-1).max() > .01
                or quaternion_distance(pose[:, 3:], pose[0, 3:]).max() > math.radians(5)
                or np.ptp(grip) > .05):
            active.append(arm)
    if not active:
        raise ValueError('Reference contains no active arm')
    return active


def default_cases():
    """A fixed small screening design, never a presumed positive/negative label."""
    cases = []
    for axis in range(3):
        for sign in (-1, 1):
            cases.append(dict(id=f'position_{axis}_{sign}', variant_family='position',
                              position_axis=axis, position_offset_m=sign*.01))
    for sign in (-1, 1):
        cases.append(dict(id=f'position_small_z_{sign}', variant_family='position',
                          position_axis=2, position_offset_m=sign*.003))
    for axis in (0, 2):
        for sign in (-1, 1):
            cases.append(dict(id=f'orientation_{axis}_{sign}', variant_family='orientation',
                              rotation_axis=axis, rotation_rad=sign*math.radians(15)))
    for shift in (-4, 4):
        cases.append(dict(id=f'gripper_shift_{shift}', variant_family='gripper_timing', gripper_shift=shift))
    for opening in (0., 1.):
        cases.append(dict(id=f'gripper_force_{int(opening)}', variant_family='gripper_timing', gripper_force=opening))
    cases.extend([
        dict(id='coupled_grasp_z_early_release', variant_family='coupled', position_axis=2,
             position_offset_m=.015, position_window='closed_phase', gripper_shift=4),
        dict(id='coupled_position_rotation', variant_family='coupled', position_axis=0,
             position_offset_m=.01, rotation_axis=2, rotation_rad=math.radians(15)),
    ])
    return cases


def intervene(reference, case):
    """Preserve frame zero and every reference index; never retime measurements."""
    result = arrays(reference)
    if case['variant_family'] == 'baseline':
        return plain_trajectory(result)
    for arm in active_arms(reference):
        pose, grip = result[arm]['pose'], result[arm]['gripper']
        indices = np.arange(1, len(pose))
        if 'position_offset_m' in case:
            selected = indices
            if case.get('position_window') == 'closed_phase':
                selected = indices[grip[indices] < .5]
            pose[selected, int(case['position_axis'])] += float(case['position_offset_m'])
        if 'rotation_rad' in case:
            half = float(case['rotation_rad'])/2
            q = np.zeros(4); q[0] = math.cos(half); q[1+int(case['rotation_axis'])] = math.sin(half)
            pose[indices, 3:] = quaternion_multiply(q, pose[indices, 3:])
        if 'gripper_shift' in case:
            original = grip.copy()
            grip[indices] = original[np.clip(indices+int(case['gripper_shift']), 0, len(grip)-1)]
        if 'gripper_force' in case:
            grip[indices] = float(case['gripper_force'])
    arrays(result)
    return plain_trajectory(result)


def command_rows(trajectory):
    data = arrays(trajectory)
    return np.concatenate((data['left']['pose'], data['left']['gripper'][:, None],
                           data['right']['pose'], data['right']['gripper'][:, None]), axis=-1)


def sample(native):
    """EEF pose is measured; gripper is the native nominal command, not force."""
    robot = native.robot
    return {arm: {'pose': list(map(float, getattr(robot, f'get_{arm}_ee_pose')())),
                  'gripper': float(getattr(robot, f'get_{arm}_gripper_val')())}
            for arm in ('left', 'right')}


def samples_to_trajectory(samples):
    return {arm: {'pose': [s[arm]['pose'] for s in samples],
                  'gripper': [s[arm]['gripper'] for s in samples]}
            for arm in ('left', 'right')}


def tracking_report(actual, commanded, *, baseline=None, reference=None, limits=None):
    limits = dict(TRACKING if limits is None else limits)
    a, c = arrays(actual), arrays(commanded)
    if len(a['left']['pose']) != len(c['left']['pose']):
        raise ValueError('Tracking requires complete command-index alignment')
    report = {}
    for arm in ('left', 'right'):
        ap, cp = a[arm]['pose'], c[arm]['pose']
        report[arm] = dict(position_m=float(np.linalg.norm(ap[1:, :3]-cp[1:, :3], axis=-1).max()),
                           rotation_rad=float(quaternion_distance(ap[1:, 3:], cp[1:, 3:]).max()),
                           gripper_fraction=float(np.max(np.abs(a[arm]['gripper'][1:]-c[arm]['gripper'][1:]))))
    realized = all(row[key] <= limits[key] for row in report.values() for key in limits)
    # Relative tracking prevents accepting ignored interventions merely because
    # they lie below the absolute tracking ceilings. Quaternion deltas are
    # expressed in the world frame, matching intervene()'s left multiplication.
    relative = {}
    if baseline is not None:
        b, r = arrays(baseline), arrays(reference)
        if any(len(value['left']['pose']) != len(a['left']['pose']) for value in (b, r)):
            raise ValueError('Relative tracking requires complete command-index alignment')
        for arm in ('left', 'right'):
            ap, cp, bp, rp = [v[arm]['pose'] for v in (a, c, b, r)]
            intended = cp[1:, :3]-rp[1:, :3]
            error = np.linalg.norm((ap[1:, :3]-bp[1:, :3])-intended, axis=-1)
            size = np.linalg.norm(intended, axis=-1)
            rotation_intended = quaternion_multiply(cp[1:, 3:], quaternion_inverse(rp[1:, 3:]))
            rotation_actual = quaternion_multiply(ap[1:, 3:], quaternion_inverse(bp[1:, 3:]))
            rotation_error = quaternion_distance(rotation_actual, rotation_intended)
            rotation_size = quaternion_distance(rotation_intended, [1., 0., 0., 0.])
            gripper_intended = c[arm]['gripper'][1:]-r[arm]['gripper'][1:]
            gripper_actual = a[arm]['gripper'][1:]-b[arm]['gripper'][1:]
            metrics = {'position_m': (error, size),
                       'rotation_rad': (rotation_error, rotation_size),
                       'gripper_fraction': (np.abs(gripper_actual-gripper_intended), np.abs(gripper_intended))}
            row = {}
            for name, (errors, sizes) in metrics.items():
                perturbed = sizes > 1e-7
                passes = bool(np.all(errors[perturbed] <= np.maximum(
                    RELATIVE_TRACKING[name], RELATIVE_TRACKING['relative_fraction']*sizes[perturbed])))
                row[name+'_error'] = float(errors.max())
                row[name+'_realized'] = passes
                realized &= passes
            relative[arm] = row
    return dict(realized=bool(realized), absolute=report, relative=relative, limits=limits,
                relative_limits=dict(RELATIVE_TRACKING),
                gripper_measurement='native_nominal_command_fraction_not_contact_force')


def replay(native, commanded):
    """Run all targets while retaining the original native success predicate.

    The native controller replans each absolute target. Its predicate is called
    at the original physics cadence, but early termination is suppressed so all
    frames are measured. Success means the unchanged predicate was ever true.
    The fixed-horizon protocol differs from a benchmark termination schedule and
    is recorded explicitly. Any planner failure remains an invalid experiment.
    """
    rows = command_rows(commanded)
    original_success = native.check_success
    original_planners = {}
    planner_results = []
    successes = []
    samples = [sample(native)]

    def predicate():
        outcome = bool(original_success())
        successes.append(outcome)
        return False

    def observe_planner(method, arm):
        def wrapped(*args, **kwargs):
            result = method(*args, **kwargs)
            planner_results.append(dict(frame=len(samples), arm=arm, status=str(result.get('status'))))
            return result
        return wrapped

    native.check_success = predicate
    for arm in ('left', 'right'):
        name = f'{arm}_plan_path'
        if hasattr(native.robot, name):
            method = getattr(native.robot, name)
            original_planners[name] = method
            setattr(native.robot, name, observe_planner(method, arm))
    native.step_lim = len(rows)-1
    native.take_action_cnt = 0
    native.eval_success = False
    error = None
    try:
        for index, row in enumerate(rows[1:], 1):
            native.take_action(row, action_type='ee')
            if int(native.take_action_cnt) != index:
                raise RuntimeError('Native command count did not advance exactly once')
            samples.append(sample(native))
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        native.check_success = original_success
        for name, method in original_planners.items():
            setattr(native.robot, name, method)
    try:
        final_success = bool(original_success())
    except Exception as exc:
        final_success = False
        error = error or f'{type(exc).__name__}: {exc}'
    complete = len(samples) == len(rows) and error is None
    return dict(actual=samples_to_trajectory(samples), native_success=bool(any(successes) or final_success),
                ever_success=bool(any(successes) or final_success), final_success=final_success,
                planner_success=(len(planner_results) == 2*(len(rows)-1)
                                 and all(p['status'] == 'Success' for p in planner_results)),
                planner_results=planner_results, execution_complete=complete, error=error,
                alignment=ALIGNMENT, native_success_rule=SUCCESS_RULE,
                early_termination_suppressed_for_complete_alignment=True)
