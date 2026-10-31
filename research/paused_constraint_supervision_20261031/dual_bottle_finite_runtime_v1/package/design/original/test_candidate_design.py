import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from candidate_design import (P, arrays, build_raw, fixed_reference, generate, json_hash,
                              load_design, marker_delta, rotation)


@pytest.fixture(scope='module')
def inputs():
    corpus = json.loads(gzip.decompress((P.parent / 'dual_bottle_revision_v1/verified_corpus.json.gz').read_bytes()))
    return corpus['refs']['10000'], load_design()


def test_budget_and_provenance(inputs):
    _, plan = inputs
    slots = plan['candidate_slots']
    assert len(slots) == 18
    assert [sum(s['category'] == key for s in slots) for key in ['model_like_crafted', 'prior_success_variation_control', 'outside_set_control']] == [12, 4, 2]
    assert all(s['source'] == 'crafted' for s in slots)
    assert 3*(1+2+len(slots)) == plan['budget']['maximum_native_episodes_including_capture'] == 63


def test_all_raw_keep_complete_reference_and_initial_state(inputs):
    ref, plan = inputs
    before = json_hash(ref)
    value = arrays(ref['trajectory'])
    for slot in plan['candidate_slots']:
        proposal, _ = build_raw(ref['trajectory'], slot)
        candidate = arrays(proposal)
        for arm in value:
            assert candidate[arm]['pose'].shape == value[arm]['pose'].shape
            np.testing.assert_array_equal(candidate[arm]['pose'][0], value[arm]['pose'][0])
            assert candidate[arm]['gripper'][0] == value[arm]['gripper'][0]
    assert json_hash(ref) == before


def test_local_y_roll_preserves_constrained_axis_and_tcp_compensation_is_explicit(inputs):
    ref, plan = inputs
    a = arrays(ref['trajectory'])
    for slot in plan['candidate_slots']:
        if slot['family'] not in {'local_y_roll', 'tcp_compensated_roll'}:
            continue
        candidate = build_raw(ref['trajectory'], slot)[0]
        for arm, value in arrays(candidate).items():
            np.testing.assert_allclose(rotation(value['pose'][:, 3:])[..., :, 1], rotation(a[arm]['pose'][:, 3:])[..., :, 1], atol=1e-14)
        if slot['family'] == 'tcp_compensated_roll':
            assert max(marker_delta(ref['trajectory'], candidate).values()) < 1e-14


def test_timewarp_anchors_are_fixed_and_gripper_never_selects_mask(inputs):
    ref, plan = inputs
    a, events = fixed_reference(ref['trajectory'])
    for slot in plan['candidate_slots']:
        if slot['family'] not in {'postclose_timewarp', 'postclose_lag'}:
            continue
        values = arrays(build_raw(ref['trajectory'], slot)[0])
        for arm in a:
            np.testing.assert_array_equal(values[arm]['pose'][[events[arm]['first_close'], -1]], a[arm]['pose'][[events[arm]['first_close'], -1]])
            np.testing.assert_array_equal(values[arm]['gripper'], a[arm]['gripper'])


@pytest.mark.parametrize('mutation', ['never_close', 'second_close', 'short_approach', 'nonfinite'])
def test_unsupported_reference_fails_without_empty_success(inputs, mutation):
    ref, _ = inputs
    value = copy.deepcopy(ref['trajectory'])
    g = value['left']['gripper']
    if mutation == 'never_close':
        value['left']['gripper'] = [1.] * len(g)
    elif mutation == 'second_close':
        g[-5:-2] = [1.] * 3
    elif mutation == 'short_approach':
        g[1:] = [0.] * (len(g)-1)
    else:
        value['left']['pose'][5][0] = float('nan')
    with pytest.raises(ValueError):
        fixed_reference(value)


def test_actual_generator_reproducible_fixed_masks_and_outside_controls(inputs):
    ref, _ = inputs
    r = generate(ref['trajectory'], 10000, ref['provenance']['scene_sha256'])
    again = generate(ref['trajectory'], 10000, ref['provenance']['scene_sha256'])
    assert r == again
    assert len(r['slots']) == 18
    for slot in r['slots']:
        attempts = slot['attempts']
        outside = slot['spec']['category'] == 'outside_set_control'
        assert [t['alpha'] for t in attempts] == ([1.] if outside else [1., .75, .5, .25, .125])
        assert all([x['id'] for x in t['score']['constraints']] == [x['id'] for x in attempts[0]['score']['constraints']] for t in attempts)
        eligible = [t for t in attempts if t['score']['commanded_constraint_pass'] != outside]
        assert slot['selected_alpha'] == (eligible[0]['alpha'] if eligible else None)
        assert slot['source'] == 'crafted' and slot['native_success'] is None
        if outside:
            assert not attempts[0]['score']['commanded_constraint_pass']


def test_no_native_launch_api_in_design():
    text = (Path(__file__).parent / 'candidate_design.py').read_text()
    # Only subprocess usage is read-only git identity validation.
    assert 'ssh' not in text and 'nvidia' not in text and 'create_task' not in text


def test_infeasible_inside_slots_are_retained_without_substitution(inputs, monkeypatch):
    import candidate_design as module
    ref, _ = inputs
    original = module.score_candidate
    def injected(reference, bundle, candidate):
        result = original(reference, bundle, candidate)
        if json_hash(reference) != json_hash(candidate):
            result['commanded_constraint_pass'] = False
        return result
    monkeypatch.setattr(module, 'score_candidate', injected)
    result = module.generate(ref['trajectory'], 10000, ref['provenance']['scene_sha256'])
    assert len(result['slots']) == 18
    for slot in result['slots'][:16]:
        assert len(slot['attempts']) == 5 and slot['raw_trajectory']
        assert slot['selected_alpha'] is None and slot['status'] == 'unavailable_no_replacement'
