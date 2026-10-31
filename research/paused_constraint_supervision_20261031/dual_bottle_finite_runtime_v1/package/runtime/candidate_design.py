"""CPU-only fixed slots. No native imports, launch functions, feedback or RNG."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
P = PACKAGE / 'design/original'
WORKTREE = PACKAGE / 'code/rax'
sys.path.insert(0, str(WORKTREE))
from services.robotwin.acceptance import json_hash  # noqa: E402
from services.robotwin.acceptance_replay import arrays, plain_trajectory, quaternion_multiply  # noqa: E402
from services.robotwin.adversarial_search import _slerp, blend, score_candidate  # noqa: E402
from services.robotwin.annotation import rotation  # noqa: E402
from scripts.validate_robotwin_constraint_acceptance import reference_bundle  # noqa: E402

PLAN_SHA256 = 'f5612e8410bc8b3dfbe1260953e31974e5d205964bbb419ce1fb32cdf97827c7'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_design():
    if sha(P / 'design_plan.json') != PLAN_SHA256:
        raise ValueError('Frozen design changed')
    plan = json.loads((P / 'design_plan.json').read_text())
    if plan['rax_commit'] != '6ba151d2f5f214907b935ca2b817728bef129d04':
        raise ValueError('Frozen source commit changed')
    receipt = json.loads((P.parent / 'dual_bottle_revision_v2/handoff_receipt.json').read_text())
    if sha(P.parent / 'dual_bottle_revision_v2/handoff_receipt.json') != plan['prior_handoff_sha256']:
        raise ValueError('Parent receipt drift')
    for file, digest in receipt['source_sha256'].items():
        if sha(WORKTREE / file) != digest:
            raise ValueError('Source hash drift: ' + file)
    for name, key in [('template.json', 'template_file_sha256'), ('comparator_A15.json', 'comparator_file_sha256')]:
        if sha(P / name) != plan[key]:
            raise ValueError('Template drift')
    return plan


def fixed_reference(reference):
    ref = arrays(reference)
    events = {}
    for arm, row in ref.items():
        g = row['gripper']
        close = np.flatnonzero((g[:-1] > .5) & (g[1:] <= .5)) + 1
        opening = np.flatnonzero((g[:-1] < .5) & (g[1:] >= .5)) + 1
        if g[0] < .99 or len(close) != 1 or len(opening):
            raise ValueError('Unsupported complete reference gripper signature: ' + arm)
        index = int(close[0])
        if index < 8 or len(g) - 1 - index < 8:
            raise ValueError('Unsupported short approach/transport reference: ' + arm)
        events[arm] = dict(first_close=index, decreasing=close.tolist(), increasing=opening.tolist())
    return ref, events


def smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0, 1)
    return x*x*(3-2*x)


def ramp_to_close(length, close):
    return smoothstep(np.arange(length) / close)


def interpolate_pose(pose, indices):
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo + 1, len(pose) - 1)
    amount = indices - lo
    output = np.empty_like(pose)
    output[:, :3] = (1-amount[:, None])*pose[lo, :3] + amount[:, None]*pose[hi, :3]
    # The frozen helper's blend amount is scalar; do not broadcast a vector into it.
    output[:, 3:] = np.stack([_slerp(pose[a, 3:], pose[b, 3:], float(t))
                             for a, b, t in zip(lo, hi, amount, strict=True)])
    return output


def build_raw(reference, slot):
    ref, events = fixed_reference(reference)
    value = copy.deepcopy(ref)
    family, params = slot['family'], slot['parameters']
    length = len(ref['left']['pose'])
    index = np.arange(length)
    for arm in ref:
        close = events[arm]['first_close']
        ramp = ramp_to_close(length, close)
        if family == 'pregrasp_open_floor':
            raw = np.minimum(ref[arm]['gripper'], params['fraction'])
            amount = smoothstep(index / params['ramp_frames'])
            value[arm]['gripper'] = (1-amount)*ref[arm]['gripper'] + amount*raw
        elif family == 'closed_floor' and arm in params['arms']:
            value[arm]['gripper'] = np.maximum(ref[arm]['gripper'], params['fraction'])
        elif family == 'local_y_roll' and arm in params['degrees']:
            half = ramp * math.radians(params['degrees'][arm]) / 2
            delta = np.column_stack([np.cos(half), np.zeros(length), np.sin(half), np.zeros(length)])
            value[arm]['pose'][:, 3:] = quaternion_multiply(ref[arm]['pose'][:, 3:], delta)
        elif family == 'closure_bump':
            fade = 1 - smoothstep((index - close) / (length - 1 - close))
            value[arm]['pose'][:, :3] += (ramp * fade)[:, None] * np.asarray(params['offset_world_m'])
        elif family in {'smooth_offset', 'step_offset'} and arm in params.get('arms', ['left', 'right']):
            amount = ramp if family == 'smooth_offset' else (index > 0).astype(float)
            value[arm]['pose'][:, :3] += amount[:, None] * np.asarray(params['offset_world_m'])
        elif family == 'tcp_compensated_roll' and arm == params['arm']:
            half = ramp * math.radians(params['degrees']) / 2
            delta = np.column_stack([np.cos(half), np.zeros(length), np.sin(half), np.zeros(length)])
            value[arm]['pose'][:, 3:] = quaternion_multiply(ref[arm]['pose'][:, 3:], delta)
            marker = np.asarray(params['offset_eef_m'])
            value[arm]['pose'][:, :3] += rotation(ref[arm]['pose'][:, 3:]) @ marker - rotation(value[arm]['pose'][:, 3:]) @ marker
        elif family in {'postclose_timewarp', 'postclose_lag'} and arm in ([params['arm']] if 'arm' in params else ref):
            u = np.clip((index - close) / (length - 1 - close), 0, 1)
            if family == 'postclose_timewarp':
                mapped = close + (length - 1 - close)*u**params['power']
            else:
                mapped = index - params['lag_frames']*np.sin(np.pi*u)
            mapped = np.where(index <= close, index, np.clip(mapped, close, length - 1))
            value[arm]['pose'] = interpolate_pose(ref[arm]['pose'], mapped)
            # Generated closure and terminal pose stay fixed. Gripper is never retimed here.
            value[arm]['pose'][[close, length-1]] = ref[arm]['pose'][[close, length-1]]
        elif family == 'gripper_shift':
            value[arm]['gripper'] = ref[arm]['gripper'][np.clip(index + params['reference_index_offset'], 0, length - 1)]
        elif family == 'gripper_force':
            value[arm]['gripper'][:] = params['fraction']
        value[arm]['pose'][0] = ref[arm]['pose'][0]
        value[arm]['gripper'][0] = ref[arm]['gripper'][0]
    result = plain_trajectory(value)
    arrays(result)
    return result, events


def marker_delta(reference, candidate):
    ref, cand = arrays(reference), arrays(candidate)
    marker = np.array([.12, 0, 0])
    return {arm: float(np.linalg.norm((cand[arm]['pose'][:, :3] + rotation(cand[arm]['pose'][:, 3:]) @ marker)
                                      - (ref[arm]['pose'][:, :3] + rotation(ref[arm]['pose'][:, 3:]) @ marker), axis=-1).max())
            for arm in ref}


def generate(reference, seed, scene_sha256):
    """Pure reference-conditioned proposal. Seed/scene identify inputs; not used as RNG."""
    plan = load_design()
    ref, events = fixed_reference(reference)
    canonical = plain_trajectory(ref)
    template = json.loads((P / 'template.json').read_text())
    bundle, audit = reference_bundle(template, canonical, seed, scene_sha256)
    before = json_hash(bundle)
    baseline = score_candidate(canonical, bundle, canonical)
    if not baseline['commanded_constraint_pass']:
        raise ValueError('Reference rejected by frozen hypothesis')
    slots = []
    for spec in plan['candidate_slots']:
        raw, _ = build_raw(canonical, spec)
        outside = spec['category'] == 'outside_set_control'
        tries = []
        for alpha in ([1.] if outside else plan['projection']['alphas']):
            candidate = blend(canonical, raw, alpha)
            score = score_candidate(canonical, bundle, candidate)
            tries.append(dict(alpha=alpha, trajectory=candidate, trajectory_sha256=json_hash(candidate),
                              score=score, tcp_max_delta_m=marker_delta(canonical, candidate)))
        allowed = [t for t in tries if t['score']['commanded_constraint_pass'] != outside]
        selected = allowed[0] if allowed else None
        slots.append(dict(spec=spec, raw_trajectory=raw, raw_trajectory_sha256=json_hash(raw),
                          attempts=tries, selected_alpha=None if selected is None else selected['alpha'],
                          selected_trajectory_sha256=None if selected is None else selected['trajectory_sha256'],
                          planned_native_eligible=selected is not None,
                          status=('outside_control_expected_rejection' if outside else 'inside_set_selected') if selected else 'unavailable_no_replacement',
                          native_success=None, source='crafted'))
    if json_hash(bundle) != before:
        raise AssertionError('Fixed reference bundle mutated')
    return dict(version=1, seed=seed, scene_sha256=scene_sha256, T=len(ref['left']['pose']),
                design_sha256=PLAN_SHA256, generator_sha256=sha(__file__), reference_sha256=json_hash(canonical),
                template_sha256=json_hash(template), bundle=bundle, bundle_sha256=before,
                reference_event_audit=events, annotation_audit=audit, slots=slots,
                source='crafted', native_started=0, measured_initial_row_preserved=True)
