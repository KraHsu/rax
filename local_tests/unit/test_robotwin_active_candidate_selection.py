"""Active-arm coverage must survive large inactive-arm novelty scores."""
import copy
import pytest
from services.robotwin.adversarial_search import select_candidates


def candidate(arm, family, axis=None, novelty=1, source='crafted'):
    params={'arm':arm}
    if axis is not None:params['local_axis']=axis
    row=dict(candidate_id=f'{source}_{arm}_{family}_{axis}',family=family,source=source,parameters=params,
             score=dict(novelty=novelty,trajectory_features={a:dict(mean_position_distance_m=0,
                        mean_rotation_distance_rad=novelty if a==arm else 0,mean_gripper_distance=0) for a in ['left','right']}))
    return row


def bank():
    rows=[candidate(a,'tool_axis_spin',i,100 if a=='right' else .01) for a in ['right','left'] for i in range(3)]
    rows += [candidate(a,f'family_{i:02d}',novelty=100 if a=='right' else .01) for a in ['right','left'] for i in range(20)]
    return rows


def test_single_active_arm_three_axes_survive_inactive_novelty_and_cap():
    audit={};rows=select_candidates(bank(),['left'],24,audit=audit)
    assert len(rows)==24
    assert audit['covered_axis_cells']==[['left',0],['left',1],['left',2]]
    assert audit['inactive_only_selected']==2
    assert audit['active_arm_loads']=={'left':22}
    assert len([c for c in rows if c['parameters']['arm']=='right'])==2


def test_dual_arm_budget_is_fair_and_reports_uncovered_cells():
    audit={};rows=select_candidates(bank(),['left','right'],24,audit=audit)
    assert len(rows)==24 and len(audit['covered_axis_cells'])==6
    assert audit['inactive_only_selected']==0
    assert abs(audit['active_arm_loads']['left']-audit['active_arm_loads']['right'])<=1
    assert audit['uncovered_cells']
    assert set(map(tuple,audit['covered_cells'])).isdisjoint(map(tuple,audit['uncovered_cells']))


def test_selection_is_deterministic_and_does_not_mutate_candidate_provenance():
    rows=bank();before=copy.deepcopy(rows);a={};b={}
    assert select_candidates(rows,['left'],24,audit=a)==select_candidates(rows,['left'],24,audit=b)
    assert a==b and rows==before


def test_model_priority_keeps_original_provenance_and_uses_inactive_cap():
    model=candidate('right','model_candidate',source='model')
    model['model_provenance']={'source_kind':'model_closed_loop','reference_blend':.5,'candidate_is_full_policy_rollout':False}
    before=copy.deepcopy(model);audit={}
    chosen=select_candidates(bank()+[model],['left'],8,audit=audit)
    assert chosen[0] is model and model==before
    assert chosen[0]['model_provenance']['candidate_is_full_policy_rollout'] is False
    assert audit['inactive_only_selected']==2 and len(chosen)==8


def test_arm_parameter_cannot_mislabel_a_nuisance_only_candidate():
    liar=candidate('right','tool_axis_spin',0);liar['parameters']['arm']='left'
    audit={};select_candidates([liar],['left'],24,audit=audit)
    assert audit['covered_axis_cells']==[] and audit['active_arm_loads']=={'left':0}
    assert audit['inactive_only_selected']==1


@pytest.mark.parametrize('budget',[1,2,3,6,12,24])
def test_never_exceeds_budget_and_inactive_cap(budget):
    audit={};rows=select_candidates(bank(),['left'],budget,audit=audit)
    assert len(rows)<=budget and audit['inactive_only_selected']<=2
