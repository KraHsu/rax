"""Counterexamples drive proposals; only independent frozen holdout can admit them."""
import copy

import numpy as np
import pytest

from services.robotwin import constraint_refinement as gate
from services.robotwin.acceptance import annotation_policy_fingerprint
from services.robotwin.programmatic_annotation import POLICY, selections, template_hash, validate_template
from test_robotwin_acceptance import reference, template


def row(seed=1, source='crafted', family='gripper_timing', success=False, accepted=True, split='dev'):
    value = dict(trial_id=f'{split}-{seed}-{source}-{family}', task='adjust_bottle', split=split,
                 scene_seed=seed, scene_sha256=f'scene-{seed}', source=source, family=family,
                 kind='candidate', native_success=success, execution_complete=True, planner_success=True,
                 realized=True, commanded_constraint_pass=accepted, actual_constraint_pass=accepted,
                 trajectory_sha256='1'*64, reference_sha256='2'*64, template_sha256=template_hash(template()),
                 annotation_policy_sha256=annotation_policy_fingerprint(), evidence_sha256='3'*64)
    if source == 'model':
        value['model_provenance'] = dict(checkpoint_sha256='4'*64, source_run_sha256='5'*64,
                                        source_kind='model_closed_loop', full_policy_rollout=True,
                                        candidate_is_full_policy_rollout=True, search_transform='none', reference_blend=1.)
    return value


def inputs(seeds=range(100, 129)):
    development = gate.development_summary([row()])
    revision = gate.gripper_phase_proposal(template(), development)
    split = dict(version=1, task='adjust_bottle', dev_seeds=[1], holdout_seeds=list(seeds),
                 holdout_generator=dict(version=1, generator_sha256='6'*64, configuration={
                     'random_seed': 42, 'candidate_budget': 4, 'baseline_repeats': 2, 'include_outside_controls': True}, sources={
                     'crafted': {'candidates_per_scene': 3},
                     'model': {'candidates_per_scene': 1, 'checkpoint_sha256': '4'*64, 'source_kind': 'model_closed_loop'}}))
    sealed, release = gate.seal_revision(revision, split, development, frozen_unix=10)
    rows = []
    for seed in seeds:
        for i in range(2):
            baseline = row(seed, 'expert', f'baseline_{i}', success=True, split='holdout')
            baseline.update(kind='baseline', template_sha256=sealed['template_sha256'])
            rows.append(baseline)
        for source, families in [('crafted', ['gripper_timing', 'position', 'stall']), ('model', ['policy'])]:
            for family in families:
                value = row(seed, source, family, success=True, split='holdout')
                value['template_sha256'] = sealed['template_sha256']
                rows.append(value)
    return sealed, release, split, development, rows


def assess(values):
    sealed, release, split, development, rows = values
    return gate.assess_holdout(sealed, release, split, development, rows, complete=True,
                               preregistered_trial_ids=[r['trial_id'] for r in rows])


def test_action_satisfying_failure_is_primary_even_if_actual_violates():
    value = row(); value['actual_constraint_pass'] = False
    assert gate.classify(value) == dict(eligible=True, false_accept=True, strong_false_accept=False, false_reject=False)
    value['planner_success'] = False
    assert not gate.classify(value)['false_accept']


def test_gripper_proposal_is_data_only_and_does_not_disable_anchors():
    dev = gate.development_summary([row()])
    proposal = gate.gripper_phase_proposal(template(), dev)
    assert proposal['state'] == 'revision_proposed'
    assert len(proposal['template']['rules']) == len(template()['rules'])+1
    added = proposal['template']['rules'][-1]
    assert added['selector']['state'] == 'all' and added['type'] == 'gripper_interval'
    assert proposal['template']['signatures'] == template()['signatures']
    bad = copy.deepcopy(proposal['patch'])
    bad['operations'] = [dict(op='replace_rule', rule=copy.deepcopy(template()['rules'][0]))]
    bad['operations'][0]['rule']['selector']['occurrence'] = 'last'
    with pytest.raises(ValueError, match='activation'):
        gate.propose_revision(template(), bad, dev)
    bad['operations'][0]['op'] = 'python'
    with pytest.raises(ValueError, match='data-only'):
        gate.propose_revision(template(), bad, dev)


def test_dev_cannot_issue_confidence_or_become_final_holdout():
    data = inputs()
    assert data[3]['confidence_eligible'] is False
    data[4][0]['scene_sha256'] = 'scene-1'
    with pytest.raises(ValueError, match='contaminated'):
        assess(data)
    split = copy.deepcopy(data[2]); split['holdout_seeds'].append(1)
    with pytest.raises(ValueError, match='overlap'):
        gate.validate_split_plan(split)


def test_exact_binomial_bound_counts_independent_scenes():
    assert gate.binomial_upper(0, 3, .95) > .6
    assert gate.binomial_upper(0, 28, .95) > .1
    assert gate.binomial_upper(0, 29, .95) < .1
    assert gate.binomial_upper(1, 29, .95) > .1
    result = assess(inputs())
    assert result['ready']
    assert result['sources']['crafted']['accepted'] == 87
    assert result['sources']['crafted']['accepted_scenes'] == 29
    assert result['sources']['crafted']['scene_failure_upper_bound'] == gate.binomial_upper(0, 29, .95)


def test_many_correlated_variants_do_not_create_independent_support():
    result = assess(inputs([100, 101, 102]))
    assert not result['ready']
    assert result['sources']['crafted']['accepted'] == 9
    assert result['sources']['crafted']['accepted_scenes'] == 3
    assert not result['sources']['crafted']['confidence_eligible']
    assert result['sources']['crafted']['scene_failure_upper_bound'] is None


def test_accepted_failures_block_and_sources_are_not_pooled():
    data = inputs(); data[-1][-1]['native_success'] = False
    result = assess(data)
    assert not result['ready']
    assert 'model_observed_accepted_failures' in result['reasons']
    assert result['sources']['crafted']['failed_accepted_scenes'] == 0
    assert result['sources']['model']['failed_accepted_scenes'] == 1


def test_rejecting_all_successful_behaviors_is_not_an_improvement():
    data = inputs()
    for value in data[-1]:
        value['commanded_constraint_pass'] = False
    result = assess(data)
    assert not result['ready']
    assert result['sources']['model']['success_retention'] == 0
    assert 'model_rejects_successful_behaviors' in result['reasons']


def test_no_post_hoc_scene_or_candidate_selection():
    data = inputs(); data[-1].pop()
    result = assess(data)
    assert not result['ready']
    assert 'model_preregistered_scene_budget_incomplete' in result['reasons']
    assert not result['sources']['model']['confidence_eligible']
    data = inputs(); data[-1][3]['scene_sha256'] = 'scene-101'
    with pytest.raises(ValueError, match='immutable scenes'):
        assess(data)


def test_holdout_cannot_hide_rejected_successes_or_omit_native_baselines():
    data = inputs()
    data[2]['holdout_generator']['configuration']['include_outside_controls'] = False
    with pytest.raises(ValueError, match='rejected controls'):
        gate.validate_generator(data[2]['holdout_generator'])
    data = inputs()
    data[-1][0]['native_success'] = False
    assert 'holdout_native_baseline_unreliable' in assess(data)['reasons']


def test_campaign_identical_slot_names_have_distinct_global_ledger_ids():
    first = gate.campaign_rows(dict(manifest_sha256='a'*64, trials=[row()]))
    second = gate.campaign_rows(dict(manifest_sha256='b'*64, trials=[row(2)]))
    assert first[0]['trial_id'] != second[0]['trial_id']
    assert first[0]['native_trial_id'] == row()['trial_id']


def test_template_and_generator_are_frozen_before_holdout():
    data = inputs(); data[0]['template']['rationale'] += ' changed'
    with pytest.raises(ValueError):
        assess(data)
    data = inputs(); data[2]['holdout_generator']['configuration']['random_seed'] += 1
    with pytest.raises(ValueError, match='Frozen'):
        assess(data)


def test_model_chunk_proxy_is_discovery_not_full_policy_success_evidence():
    data = inputs()
    model = next(r for r in data[-1] if r['source'] == 'model')
    model['model_provenance'].update(source_kind='model_receding_horizon_snapshot', full_policy_rollout=False)
    assert gate.validate_ledger_row(model)
    with pytest.raises(ValueError, match='Model source changed'):
        assess(data)
    model['model_provenance']['full_policy_rollout'] = True
    with pytest.raises(ValueError, match='distinguish'):
        gate.validate_ledger_row(model)


def test_reference_projected_full_policy_seed_is_not_a_full_policy_candidate():
    data = inputs()
    model = next(r for r in data[-1] if r['source'] == 'model')
    model['model_provenance'].update(candidate_is_full_policy_rollout=False,
                                    search_transform='fixed_index_reference_blend', reference_blend=.8)
    with pytest.raises(ValueError, match='Transformed'):
        assess(data)


def test_physical_margins_do_not_change_penalty_scale():
    t = template(); t['rules'][0]['allowed_margin'] = dict(lower=.01, upper=.04, unit='m')
    ref = reference(1, t)
    constraint = next(c for c in ref['bundle']['constraints'] if c['id'].startswith('close_'))
    assert constraint['parameters']['bounds'] == [0., .04]
    assert constraint['penalty']['scale'] == POLICY['position_margin_m']
    t['rules'][0]['allowed_margin']['unit'] = 'rad'
    with pytest.raises(ValueError, match='unit'):
        validate_template(t)


def test_partial_opening_plateaus_are_reference_fixed_and_not_drifting():
    arm = dict(grip=np.array([1., .7, .5, .5+5e-7, .5+1.1e-6, .5+1.6e-6, 0., 0.]), events=[])
    windows = selections(dict(kind='gripper_plateaus', state='all', after='episode_start'), arm)
    assert windows == [dict(mode='interval', start=2, end=4), dict(mode='interval', start=4, end=6),
                       dict(mode='interval', start=6, end=8)]


def test_requirement_marker_blocks_legacy_pilot_only_activation(tmp_path):
    from services.robotwin.collection import atomic_json
    assert gate.refinement_readiness(tmp_path, {'adjust_bottle': ['a'*64]})['ready']
    atomic_json(tmp_path/'refinement_requirement.json', dict(version=1, protocol=gate.DEFAULT_PROTOCOL))
    result = gate.refinement_readiness(tmp_path, {'adjust_bottle': ['a'*64]})
    assert result['required'] and not result['ready']
    with pytest.raises(ValueError, match='required'):
        gate.validate_frozen_gate(dict(required=False), required=True)
