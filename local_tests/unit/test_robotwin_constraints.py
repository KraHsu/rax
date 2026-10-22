import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from services.robotwin.constraint_schema import HDF5_KEY, read_embedded, validate_constraint
from services.robotwin.recording import EpisodeRecorder
from services.robotwin.session import RoboTwinSession
from services.simulator.io import RejectedBeforeAction


def template():
    root = Path(__file__).resolve().parents[2]
    c = json.loads((root/'agent_skills/control/robotwin/geometric-constraints/references/template.json').read_text())
    c['activation'] = {'mode': 'endpoint', 'start': 0, 'end': 3}
    c['evidence'].update(status='accepted', report_id='synthetic', rationale='synthetic test only')
    return c


def observation(step):
    return {'endpose': {'left_endpose': [0, 0, step*0.02, 1, 0, 0, 0], 'right_endpose': [0, 0, 0, 1, 0, 0, 0],
                       'left_gripper': 1-step*0.5, 'right_gripper': 1.},
            'joint_action': {'vector': np.zeros(14)},
            'images': {name: np.full((8, 8, 3), (200, 80, 20), dtype=np.uint8)
                       for name in ('head_camera', 'left_camera', 'right_camera')}}


def test_recording_publishes_observations_constraints_and_revisions_together(tmp_path):
    path = tmp_path/'data/episode0.hdf5'
    recorder = EpisodeRecorder(path, recording_id='test', instruction='synthetic instruction')
    c = template()
    recorder.append(observation(0)); recorder.upsert(c, 0)
    c['parameters']['bounds'] = [-0.01, 0.08]
    recorder.upsert(c, 1)
    with pytest.raises(ValueError, match='Stale'):
        recorder.upsert(c, 1)
    recorder.append(observation(1)); recorder.append(observation(2))
    assert not path.exists()
    assert recorder.finish(success=True) == str(path)
    with h5py.File(path) as h:
        b = read_embedded(h)
        assert b['constraints'][0] == c and b['provenance']['native_success'] is True
        assert len(h['endpose/left_endpose']) == len(h['observation/head_camera/rgb']) == 3
        assert h['trajectory_constraints'].attrs['revision'] == 3
        assert len(json.loads(h['trajectory_constraints/history'][()])) == 3
        assert h.attrs['instruction'] == 'synthetic instruction'


def test_failed_recording_stays_out_of_training_data(tmp_path):
    recorder = EpisodeRecorder(tmp_path/'data/episode0.hdf5', recording_id='x', instruction='x')
    recorder.append(observation(0))
    path = Path(recorder.finish(success=False))
    assert path.parent.name == 'rejected' and not list((tmp_path/'data').glob('*.hdf5'))


def test_out_of_bounds_annotation_prevents_publication(tmp_path):
    path = tmp_path/'data/episode0.hdf5'
    recorder = EpisodeRecorder(path, recording_id='x', instruction='x')
    recorder.append(observation(0)); recorder.upsert(template(), 0)
    with pytest.raises(ValueError, match='beyond'):
        recorder.finish(success=True)
    assert not path.exists()


def test_invalid_reference_event_prevents_publication(tmp_path):
    path = tmp_path/'data/episode0.hdf5'
    recorder = EpisodeRecorder(path, recording_id='x', instruction='x')
    for _ in range(3):
        recorder.append(observation(0))
    c = template()
    c['activation'].update(mode='reference_event', event={'frame': 2, 'direction': 'decreasing', 'threshold': 0.5})
    recorder.upsert(c, 0)
    with pytest.raises(ValueError, match='threshold crossing'):
        recorder.finish(success=True)
    assert not path.exists()


class Native:
    def __init__(self):
        self.take_action_cnt, self.eval_success, self.closed = 0, False, False
    def get_obs(self):
        out = observation(self.take_action_cnt)
        out['observation'] = {k: {'rgb': v, 'actor_poses': ['hidden']} for k, v in out.pop('images').items()}
        out['object_poses'] = ['hidden']
        return out
    def take_action(self, row, action_type):
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt == 2
    def close_env(self):
        self.closed = True


def test_native_session_records_and_bounds_actions_and_hides_oracle(tmp_path):
    native = Native()
    path = tmp_path/'data/episode0.hdf5'
    session = RoboTwinSession(lambda: native, task='fake', instruction='x', output=tmp_path/'operator',
                             native_step_limit=3, provenance={'collection': 0}, recording_path=path)
    reset = session.dispatch('reset', {})
    identity = {'episode_id': reset['episode_id'], 'step_id': 0}
    session.dispatch('constraint_upsert', dict(identity, constraint=template(), expected_revision=0))
    with pytest.raises(RejectedBeforeAction, match='stale'):
        session.dispatch('chunk_step', dict(identity, step_id=9, actions=[[0]*14], action_type='qpos'))
    assert native.take_action_cnt == 0
    result = session.dispatch('chunk_step', dict(identity, actions=[[0]*14]*3, action_type='qpos'))
    assert result['executed_rows'] == 2 and result['step_id'] == 2
    assert 'object_poses' not in result['observation']
    identity['step_id'] = 2
    session.dispatch('finish_pilot', identity)
    session.close(); assert native.closed and path.exists()
    with h5py.File(path) as h:
        assert len(h['endpose/left_endpose']) == 3


def test_formal_session_cannot_add_training_labels(tmp_path):
    session = RoboTwinSession(Native, task='fake', instruction='x', output=tmp_path,
                             native_step_limit=3, provenance={'collection': 1})
    reset = session.dispatch('reset', {})
    with pytest.raises(RejectedBeforeAction, match='clean training'):
        session.dispatch('constraint_upsert', dict(episode_id=reset['episode_id'], step_id=0,
                                                  constraint=template(), expected_revision=0))
    session.close()


def test_agent_template_is_valid_and_schema_matches_training_checkout_when_present():
    root = Path(__file__).resolve().parents[2]
    raw = json.loads((root/'agent_skills/control/robotwin/geometric-constraints/references/template.json').read_text())
    validate_constraint(raw)
    assert raw['evidence']['status'] == 'proposed'
    training_schema = root.parent/'OpenWAM/openwam/train/constraint_schema.py'
    if training_schema.exists():
        assert training_schema.read_bytes() == (root/'services/robotwin/constraint_schema.py').read_bytes()


def test_agent_workspace_has_constraint_template_and_no_broken_links(tmp_path):
    from services.mcp_contract import Contract
    from services.workspace import Setup, build, recipe_for
    from services.workspace.validate import broken_links, foreign_terms
    workspace = tmp_path/'workspace'
    contract = Contract(benchmark='robotwin', observation_profile='rgb-only')
    build(recipe_for(), Setup(contract, frozenset(), 'adjust_bottle'), workspace)
    assert (workspace/'.agents/skills/geometric-constraints/references/template.json').is_file()
    assert not broken_links(workspace) and not foreign_terms(workspace, 'robotwin')


def test_terminal_episode_allows_annotation_but_not_physics(tmp_path):
    from services.controller.gateway import Gateway
    from services.mcp_contract import Contract
    from services.robotwin.tools import validate_request
    native = Native()
    session = RoboTwinSession(lambda: native, task='fake', instruction='x', output=tmp_path/'sim',
        native_step_limit=3, provenance={'collection':0}, recording_path=tmp_path/'data/episode.hdf5')
    reset = session.dispatch('reset', {})
    identity = dict(episode_id=reset['episode_id'], step_id=0)
    session.dispatch('chunk_step', dict(identity, actions=[[0]*14]*2, action_type='qpos'))
    identity['step_id'] = 2
    class Backend:
        validate = staticmethod(validate_request)
        def call(self, name, args):
            assert name == 'robotwin_constraints'
            return session.dispatch('constraint_upsert', dict(identity, constraint=args['constraint'],
                                                              expected_revision=args['expected_revision'])), []
    gateway = Gateway(Backend(), None, audit=lambda _:None, charge=lambda *_:None, refund=lambda *_:None,
                      budget=lambda:{}, contract=Contract(benchmark='robotwin', observation_profile='rgb-only'))
    token = gateway.acquire(); gateway.terminal = True
    def call(name, args):
        return gateway.handle(token, {'jsonrpc':'2.0','id':1,'method':'tools/call',
                                      'params':{'name':name,'arguments':args}})
    assert 'result' in call('robotwin_constraints', dict(operation='upsert', constraint=template(), expected_revision=0))
    assert 'error' in call('robotwin_step', {'actions':[[0]*14]})
    assert native.take_action_cnt == 2
    session.dispatch('finish_pilot', identity); session.close()
