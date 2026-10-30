"""CPU proposals for finding constraint-satisfying native task failures.

This is a black-box search proposal layer, not a task-success oracle. It keeps
reference activation and observation indices fixed, evaluates the *existing*
constraint evaluator, and freezes candidates before native rollout. A candidate
that passes here is only geometrically admissible; the simulator must establish
execution validity and task failure. Model seeds are explicitly distinguished
from constructed proposals and never relabelled as policy rollouts.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from services.robotwin.acceptance import json_hash, trajectory_violations
from services.robotwin.acceptance_replay import (
    active_arms, arrays, plain_trajectory, quaternion_distance,
    quaternion_multiply,
)

SEARCH_VERSION = 'fixed_reference_redteam_v2_active_fair'
SELECTION_POLICY = {
    'version': 'active_arm_family_axis_v2',
    'reference_active_rule': {'position_m': .01, 'rotation_rad': math.radians(5), 'gripper_range': .05},
    'changed_arm_mean_threshold': {'position_m': 1e-9, 'rotation_rad': 1e-9, 'gripper_fraction': 1e-9},
    'model_reserved_max': 1, 'inactive_only_cap': 2,
    'axis_priority': [0, 1, 2],
    'allocation': 'model; active arm x spin axis; up to two distinct-family inactive controls; active family diversity; least-loaded uncovered arm-family cells; balanced active fill',
    'overflow': 'budget may leave arm-family cells uncovered; audit every uncovered cell, never increase budget',
}
FAMILY_PRIORITY = ('grasp_offset_frozen_continuation', 'grasp_offset_compensated_transport',
                   'independent_arm_delay', 'shortcut', 'stall_then_jump', 'rush_then_hold',
                   'early_close', 'never_release', 'never_close', 'gripper_chatter',
                   'detour_loop', 'reverse_segments', 'phase_reorder', 'coupled_motion_gripper',
                   'bimanual_differential', 'expert_offset', 'counterexample_neighborhood', 'tool_axis_spin')
BLEND_GRID = (1., .75, .5, .25, .125, .0625, .03125, .015625)
SOURCE_TYPES = ('expert', 'crafted', 'model')
# These normalize exploration distance, not the physical constraint boundaries.
NOVELTY_SCALES = dict(position_m=.05, rotation_rad=math.pi/4, gripper_fraction=.25)


def source_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _checked(reference, candidate):
    ref, value = arrays(reference), arrays(candidate)
    for arm in ref:
        if value[arm]['pose'].shape != ref[arm]['pose'].shape:
            raise ValueError('Candidates must retain every reference observation index; no truncation or DTW')
        a, b = value[arm], ref[arm]
        if (not np.allclose(a['pose'][0, :3], b['pose'][0, :3], rtol=0, atol=1e-10)
                or quaternion_distance(a['pose'][0, 3:], b['pose'][0, 3:]) > 1e-10
                or abs(a['gripper'][0]-b['gripper'][0]) > 1e-12):
            raise ValueError('A proposal cannot change the measured initial robot state')
        # q and -q represent one physical state. Store precisely the frozen frame
        # zero, after checking equivalence, to avoid confusing identity hashes.
        a['pose'][0] = b['pose'][0]
        a['gripper'][0] = b['gripper'][0]
    return plain_trajectory(value)


def score_candidate(reference, bundle, candidate):
    """Exact admissibility plus a max-residual search surrogate, not train loss.

    All intervals, event crossings and window eligibility come from the original
    evaluator and frozen reference. Generated gripper events never mask losses.
    """
    candidate = _checked(reference, candidate)
    measured = trajectory_violations(bundle, reference, candidate)
    constraints = {c['id']: c for c in bundle['constraints']}
    numerator, weight = 0., 0.
    for item in measured['constraints']:
        penalty = constraints[item['id']]['penalty']
        power = 2 if penalty['kind'] == 'squared_hinge' else 4
        w = penalty['weight']
        numerator += w*(item['max_violation']/penalty['scale'])**power
        weight += w
    ref, value = arrays(reference), arrays(candidate)
    features = {}
    for arm in ref:
        rp, cp = ref[arm]['pose'], value[arm]['pose']
        features[arm] = dict(
            mean_position_distance_m=float(np.linalg.norm(cp[1:, :3]-rp[1:, :3], axis=-1).mean()),
            mean_rotation_distance_rad=float(quaternion_distance(cp[1:, 3:], rp[1:, 3:]).mean()),
            mean_gripper_distance=float(np.abs(value[arm]['gripper'][1:]-ref[arm]['gripper'][1:]).mean()),
            maximum_position_step_m=float(np.linalg.norm(np.diff(cp[:, :3], axis=0), axis=-1).max()),
            maximum_rotation_step_rad=float(quaternion_distance(cp[1:, 3:], cp[:-1, 3:]).max()),
        )
    novelty = sum(row['mean_position_distance_m']/NOVELTY_SCALES['position_m']
                  + row['mean_rotation_distance_rad']/NOVELTY_SCALES['rotation_rad']
                  + row['mean_gripper_distance']/NOVELTY_SCALES['gripper_fraction']
                  for row in features.values())/2
    return dict(commanded_constraint_pass=measured['passed'],
                max_violation=measured['max_violation'],
                max_residual_surrogate=numerator/weight, novelty=float(novelty),
                constraints=measured['constraints'], trajectory_features=features,
                native_outcome='unknown_until_rollout')


def _slerp(left, right, amount):
    left, right = np.asarray(left), np.asarray(right).copy()
    dot = np.sum(left*right, axis=-1)
    right = np.where((dot < 0)[..., None], -right, right)
    dot = np.clip(np.abs(dot), 0, 1)
    angle = np.arccos(dot)
    denominator = np.sin(angle)
    safe = np.maximum(denominator, 1e-12)
    result = (np.sin((1-amount)*angle)[..., None]*left
              + np.sin(amount*angle)[..., None]*right)/safe[..., None]
    linear = (1-amount)*left+amount*right
    result = np.where((denominator < 1e-8)[..., None], linear, result)
    return result/np.linalg.norm(result, axis=-1, keepdims=True)


def blend(reference, proposed, amount):
    """Blend values at fixed indices; do not blend labels or choose activations."""
    if not math.isfinite(amount) or not 0 <= amount <= 1:
        raise ValueError('Blend amount must be in [0,1]')
    ref, value = arrays(reference), arrays(_checked(reference, proposed))
    for arm in ref:
        value[arm]['pose'][:, :3] = (1-amount)*ref[arm]['pose'][:, :3]+amount*value[arm]['pose'][:, :3]
        value[arm]['pose'][:, 3:] = _slerp(ref[arm]['pose'][:, 3:], value[arm]['pose'][:, 3:], amount)
        value[arm]['gripper'] = (1-amount)*ref[arm]['gripper']+amount*value[arm]['gripper']
    return _checked(reference, plain_trajectory(value))


def _anchors(bundle, arm, length):
    # Keyframe preservation is a proposal strategy, not a mask change: every
    # intervening frame still receives the full original constraint evaluation.
    result = {0, length-1}
    for c in bundle['constraints']:
        if c['evidence']['status'] != 'accepted' or c['subject'] != arm+'_eef':
            continue
        a = c['activation']
        if a['mode'] == 'reference_event':
            result.add(a['event']['frame'])
        elif a['mode'] == 'endpoint':
            result.add(a['end']-1)
    return sorted(result)


def _pose_anchors(value, ref, bundle):
    for arm in ref:
        for index in _anchors(bundle, arm, len(ref[arm]['pose'])):
            value[arm]['pose'][index, :3] = ref[arm]['pose'][index, :3]
    return value


def _constructed_proposals(reference, bundle, rng):
    ref = arrays(reference)
    length = len(ref['left']['pose'])
    rows = []

    def add(family, value, params=None, source='crafted'):
        for arm in ref:
            value[arm]['pose'][0] = ref[arm]['pose'][0]
            value[arm]['gripper'][0] = ref[arm]['gripper'][0]
        rows.append(dict(source=source, family=family, trajectory=plain_trajectory(value), parameters=params or {}))

    # Reference-local offsets are a control family, not the main adversary.
    for axis in range(3):
        for sign in (-1, 1):
            value = arrays(reference)
            for arm in ref:
                value[arm]['pose'][1:, axis] += sign*.01
            add('expert_offset', value, dict(axis=axis, offset_m=sign*.01), source='expert')
    for arm in ref:
        anchors = _anchors(bundle, arm, length)
        for family in ('stall_then_jump', 'rush_then_hold', 'shortcut', 'reverse_segments', 'phase_reorder'):
            value = arrays(reference)
            pose = value[arm]['pose']
            for start, end in zip(anchors, anchors[1:]):
                if end-start < 2:
                    continue
                indices = np.arange(start+1, end)
                if family == 'stall_then_jump':
                    pose[indices, :3] = ref[arm]['pose'][start, :3]
                elif family == 'rush_then_hold':
                    pose[indices, :3] = ref[arm]['pose'][end, :3]
                elif family == 'shortcut':
                    fraction = (indices-start)/(end-start)
                    pose[indices, :3] = ((1-fraction[:, None])*ref[arm]['pose'][start, :3]
                                        + fraction[:, None]*ref[arm]['pose'][end, :3])
                elif family == 'reverse_segments':
                    pose[indices, :3] = ref[arm]['pose'][indices[::-1], :3]
                else:
                    offset = max(1, len(indices)//2)
                    pose[indices, :3] = ref[arm]['pose'][np.roll(indices, offset), :3]
            add(family, value, dict(arm=arm, keyframes=anchors))
        for lag in (max(1, length//8), max(1, length//3)):
            value = arrays(reference)
            source_indices = np.maximum(np.arange(length)-lag, 0)
            value[arm]['pose'] = ref[arm]['pose'][source_indices].copy()
            _pose_anchors(value, ref, bundle)
            add('independent_arm_delay', value, dict(arm=arm, lag_frames=lag))
        for family in ('early_close', 'never_release', 'never_close', 'gripper_chatter'):
            value = arrays(reference)
            grip = value[arm]['gripper']
            if family == 'early_close':
                grip[1:] = np.minimum.accumulate(grip[1:][::-1])[::-1]
            elif family == 'never_release':
                grip[1:] = np.minimum.accumulate(grip)[1:]
            elif family == 'never_close':
                grip[1:] = 1
            else:
                grip[1:] = (np.arange(1, length) % 2).astype(float)
            add(family, value, dict(arm=arm))
        closes = np.flatnonzero((ref[arm]['gripper'][:-1] >= .5) & (ref[arm]['gripper'][1:] < .5))+1
        if len(closes):
            close = int(closes[0])
            opens = np.flatnonzero((ref[arm]['gripper'][:-1] < .5) & (ref[arm]['gripper'][1:] >= .5))+1
            releases = opens[opens > close]
            release = int(releases[0]) if len(releases) else length-1
            ramp_start = max(1, close-4)
            for axis in range(3):
                for sign in (-1, 1):
                    for family, until in [('grasp_offset_frozen_continuation', close+1),
                                          ('grasp_offset_compensated_transport', release)]:
                        value = arrays(reference)
                        envelope = np.zeros(length)
                        envelope[ramp_start:close+1] = np.linspace(1/(close-ramp_start+1), 1, close-ramp_start+1)
                        envelope[close:min(length, until+1)] = 1
                        fade_end = min(length, until+5)
                        if fade_end > until+1:
                            envelope[until+1:fade_end] = np.linspace(1, 0, fade_end-until-1)
                        value[arm]['pose'][:, axis] += sign*.02*envelope
                        add(family, value, dict(arm=arm, axis=axis, offset_m=sign*.02,
                                                close_frame=close, release_frame=release,
                                                offset_held_through_frame=until))
        for axis in range(3):
            for amplitude in (.02, .05):
                value = arrays(reference)
                phase = np.linspace(0, 2*math.pi*(1 if amplitude == .02 else 3), length)
                value[arm]['pose'][:, axis] += amplitude*np.sin(phase)
                _pose_anchors(value, ref, bundle)
                add('detour_loop', value, dict(arm=arm, axis=axis, amplitude_m=amplitude))
            for angle in (math.pi/4, math.pi/2):
                value = arrays(reference)
                q = np.zeros(4); q[0] = math.cos(angle/2); q[1+axis] = math.sin(angle/2)
                value[arm]['pose'][1:, 3:] = quaternion_multiply(value[arm]['pose'][1:, 3:], q)
                add('tool_axis_spin', value, dict(arm=arm, local_axis=axis, angle_rad=angle))
    for axis in range(3):
        value = arrays(reference)
        phase = np.sin(np.linspace(0, math.pi, length))
        value['left']['pose'][:, axis] += .03*phase
        value['right']['pose'][:, axis] -= .03*phase
        _pose_anchors(value, ref, bundle)
        add('bimanual_differential', value, dict(axis=axis, amplitude_m=.03))
    # Compound proposals can expose holes that independent one-axis edits miss.
    for index in rng.permutation(len(rows))[:12]:
        parent = rows[int(index)]
        value = arrays(parent['trajectory'])
        arm = ('left', 'right')[int(rng.integers(2))]
        value[arm]['gripper'][1:] = np.minimum.accumulate(value[arm]['gripper'])[1:]
        add('coupled_motion_gripper', value, dict(parent_family=parent['family'], parent_parameters=parent['parameters'], arm=arm))
    return rows


def _external_proposals(reference, model_candidates, feedback_parents, rng):
    proposals = []
    for item in model_candidates or []:
        provenance = item.get('model_provenance')
        if (item.get('source') != 'model' or not isinstance(provenance, dict)
                or not isinstance(provenance.get('source_kind'), str) or not provenance['source_kind']
                or any(not isinstance(provenance.get(k), str) or len(provenance[k]) != 64
                       or set(provenance[k])-set('0123456789abcdef')
                       for k in ('checkpoint_sha256', 'source_run_sha256'))):
            raise ValueError('Model proposals require explicit source kind and checkpoint/run SHA256 provenance')
        if not isinstance(item.get('candidate_id'), str) or not item['candidate_id']:
            raise ValueError('Model proposal needs a stable external candidate ID')
        proposals.append(dict(source='model', family=item.get('family', 'model_candidate'),
                              trajectory=_checked(reference, item['trajectory']), parameters=dict(external_candidate_id=item['candidate_id']),
                              model_provenance=copy.deepcopy(provenance)))
    for parent in feedback_parents or []:
        if (not isinstance(parent.get('evidence_id'), str) or not parent['evidence_id'] or parent.get('native_success') is not False
                or any(parent.get(key) is not True for key in ('execution_complete', 'planner_success', 'realized'))):
            raise ValueError('Feedback parents require an identified complete, realized native failure')
        trajectory = _checked(reference, parent['trajectory'])
        for index in range(6):
            value = arrays(trajectory)
            arm = ('left', 'right')[index % 2]
            axis = int(rng.integers(3)); amplitude = float(rng.choice([-.005, -.002, .002, .005]))
            value[arm]['pose'][1:, axis] += amplitude
            proposals.append(dict(source='crafted', family='counterexample_neighborhood', trajectory=plain_trajectory(value),
                                  parameters=dict(evidence_id=parent['evidence_id'], arm=arm, axis=axis, offset_m=amplitude)))
    return proposals


def _changed_arms(candidate):
    thresholds = SELECTION_POLICY['changed_arm_mean_threshold']
    return tuple(arm for arm, row in candidate['score']['trajectory_features'].items()
                 if (row['mean_position_distance_m'] > thresholds['position_m']
                     or row['mean_rotation_distance_rad'] > thresholds['rotation_rad']
                     or row['mean_gripper_distance'] > thresholds['gripper_fraction']))


def select_candidates(ranked, reference_active_arms, budget, *, audit=None):
    """Finite arm/family coverage before novelty; no new trajectory or source edits."""
    active = tuple(reference_active_arms)
    if not active or set(active)-{'left', 'right'} or len(active) != len(set(active)):
        raise ValueError('Explicit unique reference-active arms required')
    if type(budget) is not int or budget < 1:
        raise ValueError('Positive candidate budget required')
    rank = {c['candidate_id']: i for i, c in enumerate(ranked)}
    targets = {c['candidate_id']: tuple(a for a in _changed_arms(c) if a in active) for c in ranked}
    selected, seen, loads, inactive = [], set(), dict.fromkeys(active, 0), 0
    def choose(c):
        nonlocal inactive
        if len(selected) >= budget or c['candidate_id'] in seen:
            return False
        arms = targets[c['candidate_id']]
        if not arms and inactive >= SELECTION_POLICY['inactive_only_cap']:
            return False
        selected.append(c); seen.add(c['candidate_id'])
        if not arms: inactive += 1
        for arm in arms: loads[arm] += 1
        return True
    def available(predicate):
        return [c for c in ranked if c['candidate_id'] not in seen and predicate(c)]
    def cells():
        return {(a, c['family']) for c in selected for a in targets[c['candidate_id']]}
    models = [c for c in ranked if c['source'] == 'model']
    if models: choose(models[0])
    for axis in SELECTION_POLICY['axis_priority']:
        for arm in sorted(active, key=lambda a: (loads[a], active.index(a))):
            pool = available(lambda c: c['family'] == 'tool_axis_spin' and c['parameters'].get('local_axis') == axis
                             and c['parameters'].get('arm') == arm and arm in targets[c['candidate_id']])
            if pool: choose(pool[0])
    # Keep a small, explicit nuisance/collision-control stratum.
    control_families = {c['family'] for c in selected if not targets[c['candidate_id']]}
    for c in ranked:
        if not targets[c['candidate_id']] and c['family'] not in control_families and choose(c):
            control_families.add(c['family'])
    families = list(FAMILY_PRIORITY) + sorted({c['family'] for c in ranked}-set(FAMILY_PRIORITY))
    def fair_key(c):
        arms = targets[c['candidate_id']]
        return (min(loads[a] for a in arms), sum(loads[a] for a in arms)/len(arms), rank[c['candidate_id']])
    # Preserve global family diversity, assigning each to a least-loaded active arm.
    for family in families:
        if any(c['family'] == family and targets[c['candidate_id']] for c in selected): continue
        pool = available(lambda c: c['family'] == family and bool(targets[c['candidate_id']]))
        if pool: choose(min(pool, key=fair_key))
    # Cover remaining active-arm x family cells fairly until the fixed budget ends.
    while len(selected) < budget:
        covered = cells()
        pool = available(lambda c: any((a, c['family']) not in covered for a in targets[c['candidate_id']]))
        if not pool: break
        choose(min(pool, key=lambda c: (fair_key(c)[0], families.index(c['family']), fair_key(c)[1], rank[c['candidate_id']])))
    while len(selected) < budget:
        pool = available(lambda c: bool(targets[c['candidate_id']]))
        if not pool: break
        choose(min(pool, key=fair_key))
    if audit is not None:
        all_cells = {(a, c['family']) for c in ranked for a in targets[c['candidate_id']]}
        covered = cells()
        axis_cells = lambda rows: {(c['parameters'].get('arm'), c['parameters'].get('local_axis')) for c in rows
                                   if c['family'] == 'tool_axis_spin' and c['parameters'].get('arm') in targets[c['candidate_id']]}
        available_axes, covered_axes = axis_cells(ranked), axis_cells(selected)
        audit.update(available_axis_cells=[list(x) for x in sorted(available_axes)],
                     covered_axis_cells=[list(x) for x in sorted(covered_axes)],
                     uncovered_axis_cells=[list(x) for x in sorted(available_axes-covered_axes)],
                     policy=copy.deepcopy(SELECTION_POLICY), family_priority=list(FAMILY_PRIORITY),
                     reference_active_arms=list(active), available_candidates=len(ranked), budget=budget,
                     selected=len(selected), active_arm_loads=loads, inactive_only_selected=inactive,
                     available_cells=[list(x) for x in sorted(all_cells)],
                     covered_cells=[list(x) for x in sorted(covered)],
                     uncovered_cells=[list(x) for x in sorted(all_cells-covered)],
                     selected_candidates=[dict(candidate_id=c['candidate_id'], family=c['family'], source=c['source'],
                                               changed_arms=list(_changed_arms(c)), active_targets=list(targets[c['candidate_id']])) for c in selected])
    return selected


def generate_candidates(reference_trajectory, bundle, *, seed, budget=64, model_candidates=None, feedback_parents=None, selection_audit=None):
    """Deterministic diverse proposals, ranked toward the unchanged allowed set.

    A finite blend grid repairs proposals toward the reference. It does not
    assume the feasible region is convex or that the grid finds its boundary.
    Original model traces remain marked as model-seeded when blended. Native
    feedback only chooses next-round parents; it cannot supply a failure label
    for a newly generated candidate or modify this round's frozen constraints.
    """
    if type(seed) is not int or seed < 0 or type(budget) is not int or budget < 1:
        raise ValueError('Require a nonnegative integer seed and positive candidate budget')
    reference = _checked(reference_trajectory, reference_trajectory)
    baseline = score_candidate(reference, bundle, reference)
    if not baseline['commanded_constraint_pass']:
        raise ValueError('Frozen reference must satisfy its own constraints')
    for parent in feedback_parents or []:
        if not score_candidate(reference, bundle, parent['trajectory'])['commanded_constraint_pass']:
            raise ValueError('Feedback parent is not an inside-set failure under this frozen bundle')
    rng = np.random.Generator(np.random.PCG64(seed))
    proposals = _external_proposals(reference, model_candidates, feedback_parents, rng)
    proposals += _constructed_proposals(reference, bundle, rng)
    reference_hash, bundle_hash = json_hash(reference), json_hash(bundle)
    ranked, seen = [], set()
    for proposal in proposals:
        original = _checked(reference, proposal['trajectory'])
        evaluations = []
        for amount in BLEND_GRID:
            candidate = original if amount == 1 else blend(reference, original, amount)
            score = score_candidate(reference, bundle, candidate)
            evaluations.append((amount, candidate, score))
            if score['commanded_constraint_pass']:
                break
        feasible = [row for row in evaluations if row[2]['commanded_constraint_pass']]
        amount, candidate, score = feasible[0] if feasible else min(evaluations, key=lambda row: row[2]['max_residual_surrogate'])
        trajectory_hash = json_hash(candidate)
        if score['novelty'] <= 1e-10 or (proposal['source'], trajectory_hash) in seen:
            continue
        seen.add((proposal['source'], trajectory_hash))
        metadata = {k: copy.deepcopy(v) for k, v in proposal.items() if k != 'trajectory'}
        metadata['parameters']['reference_blend'] = amount
        if metadata['source'] == 'model':
            metadata['model_provenance']['search_transform'] = ('none' if amount == 1 else 'fixed_index_reference_blend')
            metadata['model_provenance']['reference_blend'] = amount
            metadata['model_provenance']['unmodified_model_trajectory_sha256'] = json_hash(original)
            metadata['model_provenance']['candidate_is_full_policy_rollout'] = (
                metadata['model_provenance'].get('full_policy_rollout') is True
                and metadata['model_provenance'].get('source_kind') == 'model_closed_loop'
                and amount == 1
            )
        identity = dict(version=SEARCH_VERSION, seed=seed, reference_sha256=reference_hash,
                        bundle_sha256=bundle_hash, trajectory_sha256=trajectory_hash, **metadata)
        ranked.append(dict(candidate_id='adv_'+json_hash(identity)[:20], **identity, trajectory=candidate,
                           commanded_constraint_pass=score['commanded_constraint_pass'], score=score,
                           search_version=SEARCH_VERSION))
    key = lambda c: (not c['commanded_constraint_pass'], c['score']['max_residual_surrogate'], -c['score']['novelty'], c['candidate_id'])
    ranked.sort(key=key)
    return select_candidates(ranked, active_arms(reference), budget, audit=selection_audit)


def search_manifest(reference, bundle, *, seed, budget, candidates, model_candidates=None, feedback_parents=None, selection_audit=None):
    """Small immutable search inputs/outputs receipt for a native-runner manifest."""
    reference_hash, bundle_hash = json_hash(_checked(reference, reference)), json_hash(bundle)
    if type(seed) is not int or seed < 0 or type(budget) is not int or budget < 1 or len(candidates) > budget:
        raise ValueError('Invalid frozen search seed, budget or selected count')
    ids = set()
    for candidate in candidates:
        if (candidate['candidate_id'] in ids or candidate['seed'] != seed
                or candidate['reference_sha256'] != reference_hash or candidate['bundle_sha256'] != bundle_hash
                or candidate['source'] not in SOURCE_TYPES or candidate['search_version'] != SEARCH_VERSION
                or candidate['trajectory_sha256'] != json_hash(_checked(reference, candidate['trajectory']))):
            raise ValueError('Candidate does not belong to this immutable search context')
        identity = {key: candidate[key] for key in ('version', 'seed', 'reference_sha256', 'bundle_sha256',
                    'trajectory_sha256', 'source', 'family', 'parameters')}
        if 'model_provenance' in candidate:
            identity['model_provenance'] = candidate['model_provenance']
        if candidate['candidate_id'] != 'adv_'+json_hash(identity)[:20]:
            raise ValueError('Candidate metadata changed after generation')
        fresh = score_candidate(reference, bundle, candidate['trajectory'])
        if fresh['commanded_constraint_pass'] is not candidate['commanded_constraint_pass']:
            raise ValueError('Candidate admissibility changed after generation')
        ids.add(candidate['candidate_id'])
    return dict(version=SEARCH_VERSION, search_module_sha256=source_fingerprint(), seed=seed, budget=budget,
                selection_policy=copy.deepcopy(SELECTION_POLICY), selection_audit=copy.deepcopy(selection_audit),
                bit_generator='PCG64', blend_grid=list(BLEND_GRID), novelty_scales=dict(NOVELTY_SCALES),
                reference_sha256=reference_hash, bundle_sha256=bundle_hash,
                model_inputs_sha256=json_hash(model_candidates or []), feedback_inputs_sha256=json_hash(feedback_parents or []),
                candidates=[dict(candidate_id=c['candidate_id'], source=c['source'], family=c['family'],
                                 trajectory_sha256=c['trajectory_sha256'], commanded_constraint_pass=c['commanded_constraint_pass'])
                            for c in candidates],
                meaning='Frozen finite red-team search proposals; only complete realized native outcomes can establish counterexamples.')
