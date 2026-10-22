import json
import numpy as np
import h5py
import pytest

from services.robotwin.collection import scene_fingerprint,atomic_json
from scripts.collect_robotwin_clean50 import inventory


def scene(x=0.,q=None):
    return {'actors':[{'name':'bottle','pose':[x,0.,.75,*(q or [1.,0.,0.,0.])]}],'task_articulations':[]}


def test_initial_geometry_identity_uses_pose_not_seed_or_quaternion_sign():
    assert scene_fingerprint(scene())==scene_fingerprint(scene(q=[-1.,0.,0.,0.]))
    assert scene_fingerprint(scene(.0001))==scene_fingerprint(scene())
    assert scene_fingerprint(scene(.01))!=scene_fingerprint(scene())


def test_inventory_rejects_distinct_seeds_with_duplicate_physical_states(tmp_path):
    (tmp_path/'raw').mkdir()
    for index in range(2):
        with h5py.File(tmp_path/'raw'/f'episode{index}.hdf5','w') as h:
            h.attrs['robotwin_expert_provenance']=json.dumps({'seed':index,'initial_scene_sha256':'same'})
    with pytest.raises(ValueError,match='Duplicate initial geometry'):inventory(tmp_path)


def test_inventory_counts_raw_and_published_once_and_resumes_in_order(tmp_path):
    for index,directory in [(0,'data'),(1,'raw')]:
        (tmp_path/directory).mkdir()
        with h5py.File(tmp_path/directory/f'episode{index}.hdf5','w') as h:
            h.attrs['robotwin_expert_provenance']=json.dumps({'seed':index+7,'initial_scene_sha256':str(index)})
    assert [e['seed'] for e in inventory(tmp_path)]==[7,8]
    atomic_json(tmp_path/'status.json',{'count':2})
    assert json.loads((tmp_path/'status.json').read_text())=={'count':2}


def test_queue_cannot_publish_an_episode_without_accepted_constraints(tmp_path):
    from scripts.robotwin_annotation_queue import complete
    receipt=tmp_path/'receipt.json'
    atomic_json(receipt,{'claim_id':'claim','annotation_status':'processing','path':str(tmp_path/'raw/episode0.hdf5')})
    with pytest.raises(ValueError,match='No accepted constraints'):
        complete(receipt,'claim',{'constraints':[],'source_sha256':'hash'},'agent-run')
    assert not (tmp_path/'data').exists()
    with pytest.raises(ValueError,match='Stale'):
        complete(receipt,'wrong',{},'agent-run')


def test_missing_grasp_is_retryable_but_unrelated_programming_errors_are_not():
    from scripts.collect_robotwin_clean50 import play_expert,RejectedEpisode
    class Native:
        def get_grasp_pose(self):return None
        def play_once(self):return self.get_grasp_pose()[:3]
    native=Native();original=native.get_grasp_pose
    with pytest.raises(RejectedEpisode,match='native_grasp_pose_unavailable'):play_expert(native)
    assert native.get_grasp_pose==original
    missing=None
    native.play_once=lambda:missing[0]
    with pytest.raises(TypeError):play_expert(native)
    native.play_once=lambda:native.get_grasp_pose() or 'native fallback succeeded'
    assert play_expert(native)=='native fallback succeeded'
