"""A full-policy boolean/hash string alone cannot establish original commands."""
import copy

import pytest

from services.robotwin.acceptance import file_hash, json_hash
from services.robotwin.collection import atomic_json
from services.robotwin.refinement_model_evidence import read, verify_model_evidence
from test_robotwin_constraint_refinement import row


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path/'model'
    root.mkdir()
    spec = dict(root=str(root))
    for name in ('checkpoint', 'checkpoint_receipt', 'server_config', 'server_receipt', 'collector'):
        path = tmp_path/(name+'.json')
        atomic_json(path, {'synthetic': name})
        spec[name+'_path'] = str(path)
    run = dict(task='adjust_bottle', scene_seed=1, scene_sha256='scene-1', horizon=1, created_unix=20,
               source_kind='model_closed_loop', rotation_encoding='checkpoint_legacy_reader_client_roundtrip_native_wxyz',
               alignment='reference_command_index_v1', action_contract={})
    for name in ('checkpoint', 'checkpoint_receipt', 'server_config', 'server_receipt', 'collector'):
        run[name+'_sha256'] = file_hash(spec[name+'_path'])
    atomic_json(root/'source_run.json', run)
    atomic_json(root/'configuration.json', run)
    atomic_json(root/'result.json', dict(complete=True, error=None))
    raw = [dict(frame=1, eef20=[0, 0, 0, 1, 0, 0, 0, -1, 0, .5]*2, gripper_clipped=False)]
    atomic_json(root/'raw_policy_actions.json', raw)
    atomic_json(root/'observation_hashes.json', [dict(frame=1, state=[0.]*20, camera_sha256={'head': 'a'*64})])
    trajectory = {a: dict(pose=[[0, 0, 0, 1, 0, 0, 0]]*2, gripper=[.5, .5]) for a in ('left', 'right')}
    provenance = dict(source_run_sha256=file_hash(root/'source_run.json'), checkpoint_sha256=run['checkpoint_sha256'],
                      source_kind='model_closed_loop', full_policy_rollout=True,
                      raw_policy_actions_sha256=file_hash(root/'raw_policy_actions.json'),
                      observation_hashes_sha256=file_hash(root/'observation_hashes.json'),
                      applied_action_transform=dict(gripper_clip_0_1_frames=0, time_resampling=False, expert_action_blend=False))
    candidates = {k: run[k] for k in ('task', 'scene_seed', 'scene_sha256', 'alignment', 'action_contract')}
    candidates['candidates'] = [dict(trajectory=trajectory, model_provenance=provenance)]
    atomic_json(root/'model_candidates.json', candidates)
    value = row(1, 'model', success=True)
    value.update(trajectory_sha256=json_hash(trajectory), model_provenance=dict(provenance,
                 candidate_is_full_policy_rollout=True, search_transform='none', reference_blend=1.,
                 unmodified_model_trajectory_sha256=json_hash(trajectory)))
    inventory = tmp_path/'inventory.json'
    atomic_json(inventory, dict(version=1, runs=[spec]))
    return inventory, root, value, dict(frozen_unix=10)


def test_original_commands_reconstruct_from_raw_policy_actions(evidence):
    inventory, _, row, release = evidence
    proof = verify_model_evidence(inventory, [row], release)
    assert len(proof['runs']) == 1


def test_hash_strings_cannot_replace_raw_model_proof(evidence):
    inventory, _, row, release = evidence
    row['model_provenance']['source_run_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='no auditable'):
        verify_model_evidence(inventory, [row], release)


def test_changed_raw_observation_and_checkpoint_are_detected(evidence):
    inventory, root, row, release = evidence
    atomic_json(root/'observation_hashes.json', [])
    with pytest.raises(ValueError, match='observation receipts'):
        verify_model_evidence(inventory, [row], release)


def test_rehashed_candidate_cannot_hide_an_expert_action_blend(evidence):
    inventory, root, row, release = evidence
    candidates = read(root/'model_candidates.json')
    trajectory = copy.deepcopy(candidates['candidates'][0]['trajectory'])
    trajectory['left']['pose'][1][0] = .25
    candidates['candidates'][0]['trajectory'] = trajectory
    atomic_json(root/'model_candidates.json', candidates)
    row['trajectory_sha256'] = json_hash(trajectory)
    row['model_provenance']['unmodified_model_trajectory_sha256'] = json_hash(trajectory)
    with pytest.raises(ValueError, match='position differs'):
        verify_model_evidence(inventory, [row], release)


def test_pre_freeze_model_rollout_is_not_final_holdout(evidence):
    inventory, _, row, _ = evidence
    with pytest.raises(ValueError, match='postdate'):
        verify_model_evidence(inventory, [row], dict(frozen_unix=30))
