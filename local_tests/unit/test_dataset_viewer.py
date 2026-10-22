"""Viewer geometry, immutable access, and HTTP streaming against small real HDF5s."""
import copy
import hashlib
import http.client
import io
import json
import threading
from http.server import ThreadingHTTPServer

import h5py
import numpy as np
from PIL import Image
import pytest

from services.dataset_viewer.server import Handler
from services.dataset_viewer.store import Dataset, feature_curve
from services.robotwin.annotation import reference_values
from services.robotwin.constraint_schema import HDF5_KEY, KINDS
from services.robotwin.collection import atomic_json
from test_robotwin_programmatic_annotation import episode, template
from services.robotwin.programmatic_annotation import instantiate


@pytest.fixture
def dataset(tmp_path):
    atomic_json(tmp_path/'batch_manifest.json', dict(tasks=['adjust_bottle', 'queued_task'], per_task=50,
                total_target=100, started_unix=1, updated_unix=2, status='running',
                running={'0': {'task': 'adjust_bottle', 'pid': 123}}))
    directory = tmp_path/'adjust_bottle/aloha-agilex_clean_50'
    atomic_json(directory/'collection_status.json', dict(status='recording', unique_initial_scenes=2, unique_seeds=2))
    for index, variant in enumerate(['data', 'raw']):
        path = episode(directory/variant/f'episode{index}.hdf5', seed=index, offset=index*.1)
        with h5py.File(path, 'r+') as h:
            meta = json.loads(h.attrs['robotwin_expert_provenance']); meta['native_success'] = True
            h.attrs['robotwin_expert_provenance'] = json.dumps(meta)
            h.attrs['instruction'] = 'Adjust the bottle.'
            h.create_dataset('initial_scene/json', data=json.dumps({'actors': [{'name': 'bottle', 'pose': [index*.1, 0, .8, 1, 0, 0, 0]}]}))
            if variant == 'data':
                bundle = json.loads(h[HDF5_KEY][()]); bundle['constraints'], _ = instantiate(h, template())
                h[HDF5_KEY][()] = json.dumps(bundle)
                h['trajectory_constraints/history'][()] = json.dumps([{'agent_run': 'template:fixture'}])
            # Native writer's JPEG has swapped red/blue channels; viewer restores RGB.
            image = Image.new('RGB', (80, 60), (10, 40, 220)); output = io.BytesIO(); image.save(output, 'JPEG')
            for camera in ['head_camera', 'left_camera', 'right_camera']:
                h.create_dataset(f'observation/{camera}/rgb', data=np.array([output.getvalue()]*(8+index)))
        atomic_json(directory/'annotation_queue'/f'episode{index}.json', dict(index=index, seed=index,
                    frames=8+index, initial_scene_sha256=f'scene{index}', native_success=True, annotation_status='pending'))
        video = directory/'attempts'/f'seed{index}'/'video'/f'episode{index}.mp4'
        video.parent.mkdir(parents=True); video.write_bytes(b'0123456789abcdefghij')
    return Dataset(tmp_path)


def test_live_publication_and_distinct_scenes(dataset):
    summary = dataset.summary()
    assert (summary['recorded'], summary['annotated'], summary['unique_scenes']) == (2, 1, 2)
    assert summary['tasks'][1]['status'] == 'queued'
    assert dataset.episodes('queued_task')['episodes'] == []
    assert [x['annotated'] for x in dataset.episodes('adjust_bottle')['episodes']] == [True, False]
    first = dataset.episode('adjust_bottle', 0); second = dataset.episode('adjust_bottle', 1)
    assert first['annotation_method'] == 'programmatic' and first['constraints']
    assert first['initial_scene'] != second['initial_scene'] and first['seed'] != second['seed']
    assert first['events'] == [{'arm': 'left', 'frame': 2, 'direction': 'decreasing'}]
    assert all(c['reference_violation'] == 0 for c in first['constraints'])
    assert second['annotation_method'] == 'pending' and not second['constraints']
    old = dataset.episode_path('adjust_bottle', 1); new = old.parent.parent/'data'/old.name
    old.rename(new)
    assert dataset.episode('adjust_bottle', 1)['annotated']


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('mode', ['interval', 'endpoint', 'reference_event'])
def test_all_frame_curves_match_training_feature_with_original_anchor(dataset, kind, mode):
    c = copy.deepcopy(dataset.episode('adjust_bottle', 0)['constraints'][0])
    c.update(type=kind, activation={'mode': mode, 'start': 2, 'end': 7},
             reference_frame={'origin_m': [.1, -.2, .7], 'quaternion_wxyz': [np.cos(.3), 0, 0, np.sin(.3)]},
             parameters={'bounds': [0, 1], 'axis': [0, 0, 1], 'local_axis': [1, 0, 0], 'reference_axis': [0, 1, 0]})
    if mode == 'reference_event': c['activation']['event'] = {'frame': 2}
    with h5py.File(dataset.episode_path('adjust_bottle', 0), 'r+') as h:
        poses = h['endpose/left_endpose'][:]; angles = np.linspace(0, 1.2, len(poses))
        poses[:, 3] = np.cos(angles/2); poses[:, 6] = np.sin(angles/2)
        h['endpose/left_endpose'][:] = poses
        curve = feature_curve(h, c)
        indices = [6] if mode == 'endpoint' else [2] if mode == 'reference_event' else np.arange(2, 7)
        np.testing.assert_allclose(curve[indices], reference_values(h, c), atol=1e-12)
        if kind.endswith('displacement_interval'): assert abs(curve[2]) < 1e-12


def test_frames_color_validation_and_path_confinement(dataset):
    image = Image.open(io.BytesIO(dataset.frame('adjust_bottle', 0, 3, 'head_camera')))
    assert image.size == (80, 60)
    np.testing.assert_allclose(image.getpixel((20, 20)), [220, 40, 10], atol=5)
    for args in [('adjust_bottle', 0, 9, 'head_camera'), ('adjust_bottle', 0, 0, '../data'), ('../etc', 0, 0, 'head_camera')]:
        with pytest.raises(ValueError): dataset.frame(*args)
    with pytest.raises(ValueError): dataset.episode_path('adjust_bottle', -1)
    external = dataset.root.parent/'outside.hdf5'; external.touch()
    link = dataset.directory('adjust_bottle')/'data/episode2.hdf5'; link.symlink_to(external)
    with pytest.raises(ValueError, match='Invalid episode path'): dataset.episode_path('adjust_bottle', 2)


def test_http_streams_video_and_never_mutates_artifacts(dataset):
    def fingerprint():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dataset.root.rglob('*') if p.is_file()}
    before = fingerprint()
    class TestHandler(Handler): pass
    TestHandler.dataset = dataset
    server = ThreadingHTTPServer(('127.0.0.1', 0), TestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    def get(path, method='GET', headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        connection.request(method, path, headers=headers or {})
        response = connection.getresponse(); status, result, metadata = response.status, response.read(), dict(response.getheaders())
        connection.close(); return status, result, metadata
    try:
        assert get('/')[0] == 200
        assert json.loads(get('/api/summary')[1])['recorded'] == 2
        assert json.loads(get('/api/episode?task=adjust_bottle&episode=0')[1])['constraints']
        url = '/api/video?task=adjust_bottle&episode=0'
        status, body, headers = get(url, headers={'Range': 'bytes=2-5'})
        assert (status, body, headers['Content-Range']) == (206, b'2345', 'bytes 2-5/20')
        assert get(url, headers={'Range': 'bytes=-3'})[:2] == (206, b'hij')
        assert get(url, headers={'Range': 'bytes=30-'})[0] == 416
        assert get('/api/episodes?task=../../etc')[0] == 400
        assert get('/api/summary', headers={'Host': 'untrusted.example'})[0] == 403
        assert get('/api/summary', headers={'Sec-Fetch-Site': 'cross-site'})[0] == 403
        assert get('/api/episode', method='POST')[0] == 405
        assert get('/../../etc/passwd')[0] == 404
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert fingerprint() == before
