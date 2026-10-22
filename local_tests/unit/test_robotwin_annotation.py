import json

import h5py
import numpy as np
import pytest
from PIL import Image

from test_robotwin_constraints import observation, template
from services.robotwin.annotation import digest, embed_review, export_review, reference_values
from services.robotwin.constraint_schema import read_embedded
from services.robotwin.recording import EpisodeRecorder


@pytest.fixture
def episode(tmp_path):
    path = tmp_path/'data/episode.hdf5'
    r = EpisodeRecorder(path, recording_id='fixture', instruction='Lift')
    for i in range(3):
        r.append(observation(i))
    r.finish(success=True)
    return path


def test_offline_package_preserves_frame_indices_events_and_native_colors(episode, tmp_path):
    result = export_review(episode, tmp_path/'review')
    assert result['frame_count'] == 3
    assert result['source_sha256'] == digest(episode)
    assert result['gripper_events'] == [dict(subject='left_eef', frame=1, direction='decreasing', threshold=.5)]
    pixels = np.array(Image.open(tmp_path/'review/images/000001_head_camera.jpg'))
    assert np.allclose(pixels.mean((0,1)), [200,80,20], atol=5)
    assert len((tmp_path/'review/trajectory.csv').read_text().splitlines()) == 4


def test_checked_annotations_are_embedded_with_history_and_stale_update_rejected(episode):
    c = template(); c['parameters']['bounds'] = [-.01,.08]
    proposal = {'source_sha256':digest(episode), 'constraints':[c]}
    result = embed_review(episode, proposal, agent_run='fixture-run')
    assert result['revision'] == result['accepted'] == 1
    with h5py.File(episode) as h:
        assert read_embedded(h)['constraints'] == [c]
        history=json.loads(h['trajectory_constraints/history'][()])
        assert history[0]['agent_run'] == 'fixture-run'
        assert history[0]['source_sha256'] == proposal['source_sha256']
    with pytest.raises(ValueError, match='Stale'):
        embed_review(episode, proposal, agent_run='old-run')


@pytest.mark.parametrize('invalid', ['bounds','event','frame'])
def test_bad_accepted_annotations_cannot_change_episode(episode, invalid):
    c=template(); c['parameters']['bounds']=[-.01,.08]
    if invalid=='bounds': c['parameters']['bounds']=[.2,.3]
    if invalid=='event':
        c['activation'].update(mode='reference_event', event={'frame':2,'direction':'decreasing','threshold':.5})
    if invalid=='frame': c['activation']['end']=4
    before=digest(episode)
    with pytest.raises(ValueError):
        embed_review(episode, {'source_sha256':before,'constraints':[c]}, agent_run='bad')
    assert digest(episode)==before


def test_reference_geometry_uses_physical_wxyz_and_fixed_frame(episode):
    c=template(); c['type']='axis_angle_interval'
    c['parameters']={'local_axis':[1,0,0], 'reference_axis':[0,1,0], 'bounds':[0,np.pi]}
    with h5py.File(episode) as h:
        assert np.allclose(reference_values(h,c), np.pi/2)
        c['reference_frame']['quaternion_wxyz']=[np.sqrt(.5),0,0,-np.sqrt(.5)]
        assert np.allclose(reference_values(h,c), 0, atol=1e-7)
        c['type']='axis_displacement_interval'; c['parameters']={'axis':[0,0,1], 'bounds':[0,1]}
        assert np.allclose(reference_values(h,c), .04)


def test_native_scene_metadata_retains_each_episode_and_active_arm(tmp_path):
    from scripts.collect_robotwin_expert import publish_scene_info
    publish_scene_info(tmp_path, 0, {'info':{'active_arm':'left'}})
    publish_scene_info(tmp_path, 1, {'info':{'active_arm':'right'}})
    assert json.loads((tmp_path/'scene_info.json').read_text()) == {
        'episode_0':{'info':{'active_arm':'left'}}, 'episode_1':{'info':{'active_arm':'right'}}}
