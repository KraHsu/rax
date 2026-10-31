"""Fixed-reference branch admission for finite native development, never model events.

The frozen reference file and separate receipt are bound by byte hashes. Events
and frame count are recomputed from complete numeric arrays on every verification.
Dustbin initial-side prediction is diagnostic; other static/reference mismatches
remain outside the reviewed domain. This module does not publish or alter labels.
"""
import hashlib
import json
import math
from pathlib import Path

from .acceptance_replay import arrays
from .collection import atomic_json, scene_fingerprint
from .programmatic_annotation import template_hash, validate_template

POLICY = 'sealed_complete_reference_branch_v1'


class BranchIntegrityError(ValueError):
    """Infrastructure/identity failure: stop the run, never select another seed."""


class ReferenceBranchMismatch(ValueError):
    """Legitimate reference outside this fixed branch; preserve bounded rejection."""


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_fingerprint():
    return dict(policy=POLICY, source_sha256=file_hash(__file__))

SOURCE_SHA256 = {
    'stack_bowls_two': '79d0dd8099fae8a9c8caa82423b4f19ef1604158d84a1abb22555819707ca436',
    'place_bread_basket': '3751aa10c991f638be145f7536c9e1e398b1e116ba81f207dd4a11a6ae6a99ab',
    'place_mouse_pad': '47e8aea42824552756a74e41b4aeb2fb271336d50ad1c9d816d0b9f5b3103fb1',
    'stack_bowls_three': 'aa2cbd2f066f248128825a0a677600340a875f100eae55bbd19173352e46a17b',
    'put_bottles_dustbin': '641cf31132780f0d05fcf0fea4106ed794e2aa6dc35ed49daa4517e19e3ec7dd',
}

BRANCHES = {
    ('stack_bowls_two', 'split_1_1'): ('0659bafd4ede8f388f25fca5dd9778a1d254bd105220b9b70eec11257f8e6a6a', 'active'),
    ('stack_bowls_two', 'same_arm_2'): ('22fc642b97263113ceafb13effce1404c53501826651e690d0f8f274b0f77172', 'active'),
    ('place_bread_basket', 'single'): ('5ff81b5d25d05b7137d64144503a398f4fbf48d83beec8dd260cc7b54e1cd115', 'active'),
    ('place_bread_basket', 'split_1_1'): ('abf1a3a43c4b6148beeb2f291004feb3f0fa891f4bb8f1a56e1cef0d3667bae4', 'active'),
    ('place_bread_basket', 'same_arm_2'): ('fa059d4bb0fe1736473379f849bfc784f4764e3cb9ad1c70c447eb9f18748351', 'active'),
    ('place_mouse_pad', 'right'): ('7775009c3d41eb53386ce4187661905f2b2f3e41dbd24c04ad952f8ebb8f1210', 'active'),
    ('place_mouse_pad', 'left'): ('ce13e86108ef2f7afb165450715e3f6456a4ebff889b995b92623eec9d0df3d1', 'active'),
    ('stack_bowls_three', 'split_1_2'): ('459c83de03d73d509c675a46802314feba4f192a9b688dcfaa15ab3bdda34b0b', 'active'),
    ('stack_bowls_three', 'same_arm_3'): ('ee15ca90d7ea2c813bd6f76598070da59cd3731180456a8745a946ade7ddb513', 'candidate'),
    ('put_bottles_dustbin', 'right_1'): ('6ad04e9f97ed72c13916ac097fb7a79882935882a78746bb5b56a954820ddc17', 'active'),
    ('put_bottles_dustbin', 'right_2'): ('86a81e6aebc0b79626cf93f8eb9431f8b43e82c456cc217a1d20a4ed48f5eb45', 'active'),
    ('put_bottles_dustbin', 'right_3'): ('37a82c47aee978b2c86f51278d75201fdcf7c51df9b19ba7d9965b782e018475', 'candidate'),
    ('put_bottles_dustbin', 'right_0'): (None, 'missing'),
}


def static_branch(task, scene, source_sha256):
    if task not in SOURCE_SHA256 or source_sha256 != SOURCE_SHA256[task]:
        raise ValueError('unsupported task or native source drift')
    actor_name, allowed_counts = {
        'stack_bowls_two': ('002_bowl', (2,)),
        'stack_bowls_three': ('002_bowl', (3,)),
        'place_bread_basket': ('075_bread', (1, 2)),
        'place_mouse_pad': ('047_mouse', (1,)),
        'put_bottles_dustbin': ('114_bottle', (3,)),
    }[task]
    actors = [a for a in scene['actors'] if a['name'] == actor_name]
    if len(actors) not in allowed_counts:
        raise ValueError('unexpected object count; empty bread is not a valid task reference')
    for actor in actors:
        pose = actor['pose']
        if len(pose) != 7 or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in pose):
            raise ValueError('invalid initial actor pose')
        if pose[0] == 0:
            raise ValueError('zero-X actor is outside this audited selector domain')
    counts = {arm: sum(('right' if a['pose'][0] > 0 else 'left') == arm for a in actors)
              for arm in ('left', 'right')}
    if task == 'put_bottles_dustbin':
        key = 'right_' + str(counts['right'])
        expected = {'left': 3, 'right': counts['right']}
    elif task == 'place_mouse_pad':
        key = 'right' if counts['right'] else 'left'
        expected = counts
    elif task == 'place_bread_basket' and len(actors) == 1:
        key, expected = 'single', counts
    elif 0 in counts.values():
        key, expected = 'same_arm_' + str(len(actors)), counts
    else:
        key, expected = ('split_1_1' if len(actors) == 2 else 'split_1_2'), counts
    th, status = BRANCHES[(task, key)]
    return dict(task=task, branch_id=key, template_sha256=th, registry_status=status,
                initial_actor_name=actor_name, initial_actors=actors,
                initial_side_counts=counts, expected_reference_cycles=expected,
                source_sha256=source_sha256)


def select_fixed_reference(task, scene, reference_gripper_events, source_sha256):
    """Strictly verify the immutable reference's per-arm events against static source.

    No model action argument exists. This reports candidate/missing applicability;
    it NEVER returns those as an active template. scene_info is deliberately absent.
    """
    result = static_branch(task, scene, source_sha256)
    if set(reference_gripper_events) != {'left', 'right'}:
        raise ValueError('both reference arms must be present')
    sequences = {}
    cycles = {}
    for arm in ('left', 'right'):
        events = reference_gripper_events[arm]
        previous = -1
        for event in events:
            frame = event['frame']
            if not isinstance(frame, int) or isinstance(frame, bool) or frame <= previous or frame < 1:
                raise ValueError('reference event frames must be positive and strictly increasing')
            previous = frame
        sequence = [e['event'] for e in events]
        if len(sequence) % 2 or sequence != ['close', 'open'] * (len(sequence) // 2):
            raise ValueError('invalid reference close/open sequence')
        cycles[arm] = len(sequence) // 2
        sequences[arm] = ','.join(sequence)
    result['static_prediction_matches_reference'] = cycles == result['expected_reference_cycles']
    result['static_predicted_branch_id'] = result['branch_id']
    if task == 'put_bottles_dustbin':
        if cycles['left'] != 3 or not 0 <= cycles['right'] <= 3:
            raise ValueError('reference cycles violate three-bottle native expert structure')
        key = 'right_' + str(cycles['right'])
        th, status = BRANCHES[(task, key)]
        result.update(branch_id=key, template_sha256=th, registry_status=status,
                      selection_basis='fixed_reference_per_arm_cycles',
                      static_prediction_scope='initial-side prediction only; native re-reads X at each pickup')
    else:
        if not result['static_prediction_matches_reference']:
            raise ValueError('static source/reference arm-cycle disagreement; no fallback template')
        result['selection_basis'] = 'initial_scene_and_fixed_reference_agreement'
    result['actual_reference_cycles'] = cycles
    result['reference_signature'] = sorted(v for v in sequences.values() if v)
    result['reference_arm_sequences'] = sequences
    result['active_template_sha256'] = result['template_sha256'] if result['registry_status'] == 'active' else None
    result['training_or_native_acceptance'] = False
    return result


def validate_branch_spec(spec, template):
    required = {'version', 'policy', 'task', 'branch_id', 'template_sha256',
                'native_source_sha256', 'registry_path', 'registry_file_sha256',
                'registry_entry', 'seed_start', 'seeds', 'max_seed_attempts', 'baseline_repeats'}
    if not isinstance(spec, dict) or set(spec) != required or spec['version'] != 1 or spec['policy'] != POLICY:
        raise BranchIntegrityError('Wrong fixed-reference branch specification')
    key = (spec['task'], spec['branch_id'])
    if key not in BRANCHES or BRANCHES[key][1] != 'active':
        raise BranchIntegrityError('Only the ten reviewed original active branches are admitted')
    if spec['template_sha256'] != BRANCHES[key][0] or spec['native_source_sha256'] != SOURCE_SHA256[spec['task']]:
        raise BranchIntegrityError('Branch template/native source identity changed')
    try:
        validate_template(template)
    except (ValueError, TypeError, KeyError) as exc:
        raise BranchIntegrityError(f'Malformed frozen branch template: {exc}') from exc
    entry = spec['registry_entry']
    if (not isinstance(entry, dict) or entry.get('status') != 'active'
            or entry.get('template') != template or template['task'] != spec['task']
            or template_hash(template) != spec['template_sha256']):
        raise BranchIntegrityError('Registry status/template/subject contamination')
    for key in ('registry_file_sha256', 'native_source_sha256', 'template_sha256'):
        value = spec[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise BranchIntegrityError('Malformed source hash')
    if not isinstance(spec['registry_path'], str) or not Path(spec['registry_path']).is_absolute():
        raise BranchIntegrityError('Registry source must have an explicit absolute path')
    keys = [branch for (task, branch), (_, status) in BRANCHES.items() if task == spec['task'] and status == 'active']
    expected_start = 10000 + 100 * keys.index(spec['branch_id'])
    if any(type(spec[k]) is not int for k in ('seed_start', 'seeds', 'max_seed_attempts', 'baseline_repeats')):
        raise BranchIntegrityError('Finite capture bounds must be integers')
    if [spec[k] for k in ('seed_start', 'seeds', 'max_seed_attempts', 'baseline_repeats')] != [expected_start, 3, 30, 2]:
        raise BranchIntegrityError('Branch seed interval or finite budget changed')
    return spec


def validate_external_sources(spec, template, native_root):
    validate_branch_spec(spec, template)
    if file_hash(Path(native_root) / 'envs' / (spec['task'] + '.py')) != spec['native_source_sha256']:
        raise BranchIntegrityError('Native branch source drift')
    path = Path(spec['registry_path'])
    if file_hash(path) != spec['registry_file_sha256'] or json.loads(path.read_text()) != spec['registry_entry']:
        raise BranchIntegrityError('Original active registry input drift')


def reference_admission(reference, spec, template):
    """Derive all branch evidence from complete arrays; no caller event table accepted."""
    validate_branch_spec(spec, template)
    try:
        trajectory = reference['trajectory']
        if set(trajectory) != {'left', 'right'} or any(set(trajectory[a]) != {'pose', 'gripper'} for a in trajectory):
            raise ValueError('Unexpected trajectory metadata/event fields')
        data = arrays(trajectory)
        provenance, scene = reference['provenance'], reference['initial_scene']
        if set(provenance) != {'task', 'seed', 'scene_sha256', 'mode', 'split', 'source_episode_sha256', 'recording_method'}:
            raise ValueError('Unexpected role/event metadata in reference provenance')
        count = len(data['left']['gripper'])
        seed = provenance['seed']
        if type(seed) is not int or not spec['seed_start'] <= seed < spec['seed_start'] + spec['max_seed_attempts']:
            raise ValueError('Reference seed is outside the frozen branch scan')
        if provenance['task'] != spec['task'] or provenance['mode'] != 'demo_clean' or provenance['split'] != 'validation':
            raise ValueError('Reference task or development role differs')
        source_hash = canonical_hash(dict(scene=scene, trajectory=trajectory))
        scene_hash = scene_fingerprint(scene)
        if provenance['source_episode_sha256'] != source_hash or provenance['scene_sha256'] != scene_hash:
            raise ValueError('Reference content/scene hash mismatch')
        if provenance.get('recording_method') != 'native_expert_save_freq_trace_no_images':
            raise ValueError('Only captured complete native expert reference is admitted')
        events = {}
        for arm, row in data.items():
            grip = row['gripper']
            if grip[0] <= .5 or grip[-1] <= .5:
                raise ReferenceBranchMismatch('Reference must start and finish on the open side of threshold')
            events[arm] = [dict(frame=i, event='close' if grip[i-1] > .5 else 'open')
                for i in range(1, count)
                if (grip[i-1] > .5 and grip[i] <= .5) or (grip[i-1] < .5 and grip[i] >= .5)]
        # A caller cannot override freshly derived events by adding metadata fields.
        if any(k in reference for k in ('reference_gripper_events', 'branch_id', 'active_arm', 'T')):
            raise ValueError('Caller-provided branch/event/frame metadata is forbidden')
    except ReferenceBranchMismatch:
        raise
    except Exception as exc:
        raise BranchIntegrityError(f'Malformed or unbound complete reference: {exc}') from exc
    try:
        selection = select_fixed_reference(spec['task'], scene, events, spec['native_source_sha256'])
    except ValueError as exc:
        raise ReferenceBranchMismatch(str(exc)) from exc
    if selection['active_template_sha256'] != spec['template_sha256'] or selection['branch_id'] != spec['branch_id']:
        raise ReferenceBranchMismatch('Fixed reference belongs to another or unsupported branch')
    return dict(version=1, policy=policy_fingerprint(), branch_spec_sha256=canonical_hash(spec),
                task=spec['task'], branch_id=spec['branch_id'], template_sha256=spec['template_sha256'],
                seed=seed, total_frames=count, action_frames=count-1, initial_state_frame=0,
                source_episode_sha256=source_hash, scene_sha256=scene_hash,
                reference_events=events, selection=selection)


def seal_reference(reference_path, receipt_path, spec, template):
    """Seal a sidecar AFTER reference bytes exist; no self-referential hash."""
    path, sidecar = Path(reference_path), Path(receipt_path)
    if sidecar.exists():
        raise BranchIntegrityError('Existing branch receipt is immutable')
    raw = path.read_bytes()
    reference = json.loads(raw)
    receipt = reference_admission(reference, spec, template)
    receipt['reference_file_sha256'] = hashlib.sha256(raw).hexdigest()
    if path.read_bytes() != raw:
        raise BranchIntegrityError('Reference changed during sealing')
    atomic_json(sidecar, receipt)
    return receipt


def verify_reference(reference_path, receipt_path, spec, template, *, expected_reference_sha256, expected_receipt_sha256):
    path, sidecar = Path(reference_path), Path(receipt_path)
    raw, receipt_raw = path.read_bytes(), sidecar.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_reference_sha256 or hashlib.sha256(receipt_raw).hexdigest() != expected_receipt_sha256:
        raise BranchIntegrityError('Sealed reference/receipt byte hash mismatch')
    try:
        reference, receipt = json.loads(raw), json.loads(receipt_raw)
        expected = reference_admission(reference, spec, template)
        expected['reference_file_sha256'] = expected_reference_sha256
        if receipt != expected:
            raise ValueError('Receipt events, T, source identity or branch differs from complete reference')
    except ReferenceBranchMismatch as exc:
        raise BranchIntegrityError('Previously admitted reference no longer matches its branch') from exc
    except (ValueError, TypeError, KeyError) as exc:
        raise BranchIntegrityError(str(exc)) from exc
    return receipt
