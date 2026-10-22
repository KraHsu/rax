"""Bounded, read-only access to published and pending trajectory artifacts."""
from collections import OrderedDict
import copy
import io
import json
from pathlib import Path
import re
import threading
import time

import h5py
import numpy as np
from PIL import Image

from services.robotwin.annotation import reference_values, rotation
from services.robotwin.constraint_schema import read_embedded


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def feature_curve(handle, constraint):
    """Plot a primitive over the whole reference, keeping its original anchor."""
    c = constraint; arm = c['subject'].split('_')[0]; p = c['parameters']
    pose = np.asarray(handle[f'endpose/{arm}_endpose']); frame = c['reference_frame']
    basis = rotation(frame['quaternion_wxyz']); kind = c['type']
    if kind == 'gripper_interval': return np.asarray(handle[f'endpose/{arm}_gripper']).reshape(-1)
    if kind == 'axis_angle_interval':
        axes = rotation(pose[:, 3:]) @ np.asarray(p['local_axis']); target = basis @ np.asarray(p['reference_axis'])
        return np.arctan2(np.linalg.norm(np.cross(axes, target), axis=-1), (axes*target).sum(-1))
    if kind == 'angular_displacement_interval':
        r = rotation(pose[c['activation']['start'], 3:]).T @ rotation(pose[:, 3:])
        skew = np.stack((r[:,2,1]-r[:,1,2], r[:,0,2]-r[:,2,0], r[:,1,0]-r[:,0,1]), -1)
        return np.arctan2(np.linalg.norm(skew, axis=-1)*.5, (np.trace(r, axis1=-2, axis2=-1)-1)*.5)
    origin = pose[c['activation']['start'], :3] if kind == 'axis_displacement_interval' else np.asarray(frame['origin_m'])
    delta = pose[:, :3]-origin
    if kind == 'point_distance_interval': return np.linalg.norm(delta, axis=-1)
    axis = basis @ np.asarray(p['axis']); projected = delta @ axis
    if kind == 'radial_distance_interval': return np.linalg.norm(delta-projected[:, None]*axis, axis=-1)
    return projected


class Dataset:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.manifest = read_json(self.root/'batch_manifest.json')
        if not self.manifest: raise ValueError('Dataset batch manifest is missing')
        self.tasks = self.manifest['tasks']
        if any(not re.fullmatch(r'[a-z0-9_]+', t) for t in self.tasks): raise ValueError('Invalid task manifest')
        self.lock = threading.Lock(); self.cache = OrderedDict()

    def cached(self, key, ttl, load):
        now = time.monotonic()
        with self.lock:
            if key in self.cache and now-self.cache[key][0] < ttl:
                self.cache.move_to_end(key); return self.cache[key][1]
        value = load()
        with self.lock:
            self.cache[key] = (now, value); self.cache.move_to_end(key)
            while len(self.cache) > 192: self.cache.popitem(last=False)
        return value

    def directory(self, task):
        if task not in self.tasks: raise ValueError('Unknown task')
        path = (self.root/task/'aloha-agilex_clean_50').resolve()
        if not path.is_relative_to(self.root): raise ValueError('Invalid dataset path')
        return path

    def episode_path(self, task, index):
        if type(index) is not int or not 0 <= index < self.manifest['per_task']: raise ValueError('Invalid episode index')
        directory = self.directory(task)
        for variant in ('data', 'raw'):
            path = (directory/variant/f'episode{index}.hdf5').resolve()
            if not path.is_relative_to(self.root): raise ValueError('Invalid episode path')
            if path.is_file(): return path
        raise FileNotFoundError('Trajectory is not recorded yet')

    def summary(self):
        def load():
            manifest = read_json(self.root/'batch_manifest.json'); rows = []
            templates = {}
            for path in (self.root/'annotation_templates').glob('*/*.json'):
                entry = read_json(path)
                if entry.get('status') == 'active':
                    task = entry['template']['task']; templates[task] = templates.get(task, 0)+1
            running = {v['task']: int(gpu) for gpu, v in manifest.get('running', {}).items()}
            for task in self.tasks:
                directory = self.directory(task); state = read_json(directory/'collection_status.json', {})
                data = {p.name for p in (directory/'data').glob('episode*.hdf5')}
                raw = {p.name for p in (directory/'raw').glob('episode*.hdf5')}
                count = len(data | raw); status = state.get('status', 'queued')
                if status == 'failed' and task in manifest.get('pending_tasks', []): status = 'retry_queued'
                rows.append({'id': task, 'recorded': count, 'annotated': len(data), 'target': manifest['per_task'],
                             'status': status, 'gpu': running.get(task), 'templates': templates.get(task, 0),
                             'unique_scenes': state.get('unique_initial_scenes', 0), 'unique_seeds': state.get('unique_seeds', 0),
                             'active_seed': state.get('active_seed'), 'error': state.get('error') if status == 'failed' else None})
            return {'dataset': self.root.name, 'updated_unix': time.time(), 'started_unix': manifest['started_unix'],
                    'heartbeat_unix': manifest.get('updated_unix'), 'batch_status': manifest['status'],
                    'target': manifest['total_target'], 'recorded': sum(t['recorded'] for t in rows),
                    'annotated': sum(t['annotated'] for t in rows), 'ready_tasks': sum(t['annotated'] == t['target'] for t in rows),
                    'unique_scenes': sum(t['unique_scenes'] for t in rows), 'running_count': len(running), 'tasks': rows}
        return self.cached(('summary',), 4, load)

    def episodes(self, task):
        directory = self.directory(task)
        def load():
            result = []
            for path in (directory/'annotation_queue').glob('episode*.json'):
                r = read_json(path); index = int(r['index'])
                # A moved HDF5 can precede the queue receipt's final atomic update.
                published = (directory/'data'/f'episode{index}.hdf5').exists()
                result.append({'index': index, 'seed': r['seed'], 'frames': r['frames'], 'annotated': published,
                               'constraints': r.get('annotation_result', {}).get('accepted'),
                               'initial_sha256': r['initial_scene_sha256'], 'native_success': r['native_success'],
                               'annotation_status': 'done' if published else r['annotation_status'],
                               'recording_seconds': sum(r.get('seconds', {}).values())})
            return {'task': task, 'episodes': sorted(result, key=lambda r: r['index'])}
        return self.cached(('episodes', task), 4, load)

    def episode(self, task, index):
        path = self.episode_path(task, index); key = ('episode', str(path), path.stat().st_mtime_ns)
        def load():
            with h5py.File(path, 'r') as h:
                bundle = read_embedded(h); meta = json.loads(h.attrs['robotwin_expert_provenance'])
                length = len(h['endpose/left_endpose']); arms = {}; events = []
                for arm in ('left', 'right'):
                    pose = np.asarray(h[f'endpose/{arm}_endpose']); grip = np.asarray(h[f'endpose/{arm}_gripper']).reshape(-1)
                    arms[arm] = {'pose': pose.tolist(), 'gripper': grip.tolist()}
                    for direction, sign in (('decreasing', -1), ('increasing', 1)):
                        for f in np.flatnonzero((sign*(grip[:-1]-.5) < 0) & (sign*(grip[1:]-.5) >= 0))+1:
                            events.append({'arm': arm, 'frame': int(f), 'direction': direction})
                constraints = []
                for source in bundle['constraints']:
                    c = copy.deepcopy(source); a = c['activation']
                    indices = list(range(a['start'], a['end']))
                    if a['mode'] == 'endpoint': indices = indices[-1:]
                    elif a['mode'] == 'reference_event': indices = [a['event']['frame']]
                    values = feature_curve(h, c); active = reference_values(h, c)
                    lo, hi = c['parameters']['bounds']
                    c.update(values=values.tolist(), active_frames=indices,
                             unit='fraction' if c['type'] == 'gripper_interval' else 'rad' if 'angle' in c['type'] or 'angular' in c['type'] else 'm',
                             reference_violation=float(np.maximum(lo-active, 0).max()+np.maximum(active-hi, 0).max()))
                    constraints.append(c)
                scene = json.loads(h['initial_scene/json'][()]) if 'initial_scene/json' in h else None
                history = json.loads(h['trajectory_constraints/history'][()]) if 'trajectory_constraints/history' in h else []
                return {'task': task, 'index': index, 'seed': meta['seed'], 'frames': length, 'annotated': path.parent.name == 'data',
                        'instruction': str(h.attrs.get('instruction', '')), 'native_success': meta['native_success'],
                        'initial_sha256': meta['initial_scene_sha256'], 'initial_scene': scene, 'provenance': meta,
                        'action_contract': bundle['action_contract'], 'arms': arms, 'events': sorted(events, key=lambda x: x['frame']),
                        'constraints': constraints, 'cameras': [c for c in h['observation'] if 'rgb' in h['observation'][c]],
                        'revision': int(h['trajectory_constraints'].attrs.get('revision', 0)),
                        'annotation_method': 'programmatic' if history and history[-1].get('agent_run', '').startswith('template:') else 'rax' if history else 'pending',
                        'video': self.video_path(task, index, meta['seed'], required=False) is not None,
                        'recording_id': bundle['provenance']['recording_id']}
        return self.cached(key, 300, load)

    def frame(self, task, index, frame, camera, width=640):
        if camera not in ('head_camera', 'front_camera', 'left_camera', 'right_camera'): raise ValueError('Unknown camera')
        if not 64 <= width <= 960 or frame < 0: raise ValueError('Invalid frame request')
        path = self.episode_path(task, index)
        # Image bytes are immutable across constraint revisions and publication.
        def load():
            with h5py.File(self.episode_path(task, index), 'r') as h:
                dataset = h[f'observation/{camera}/rgb']
                if frame >= len(dataset): raise ValueError('Frame outside trajectory')
                raw = np.asarray(dataset[frame]).tobytes()
            pixels = np.asarray(Image.open(io.BytesIO(raw)).convert('RGB'))[..., ::-1]
            image = Image.fromarray(pixels); image.thumbnail((width, width))
            output = io.BytesIO(); image.save(output, 'JPEG', quality=88)
            return output.getvalue()
        return self.cached(('frame', task, index, frame, camera, width), 600, load)

    def video_path(self, task, index, seed=None, required=True):
        if seed is None:
            with h5py.File(self.episode_path(task, index), 'r') as h: seed = json.loads(h.attrs['robotwin_expert_provenance'])['seed']
        path = (self.directory(task)/'attempts'/f'seed{int(seed)}'/'video'/f'episode{index}.mp4').resolve()
        if not path.is_relative_to(self.root): raise ValueError('Invalid video path')
        if path.is_file(): return path
        if required: raise FileNotFoundError('Native video is unavailable')
        return None
