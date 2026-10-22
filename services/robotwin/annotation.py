"""Small offline review packages and checked, atomic HDF5 annotation updates."""
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
from PIL import Image

from services.robotwin.constraint_schema import HDF5_KEY, canonical_json, read_embedded, validate_bundle, validate_reference_events


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-4):
        raise ValueError('Invalid measured unit WXYZ quaternion')
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def reference_values(handle, constraint):
    c = constraint; a = c['activation']; p = c['parameters']
    arm = c['subject'].split('_')[0]
    pose = np.asarray(handle[f'endpose/{arm}_endpose'])
    grip = np.asarray(handle[f'endpose/{arm}_gripper']).reshape(-1)
    indices = np.arange(a['start'], a['end'])
    if a['mode'] == 'endpoint':
        indices = indices[-1:]
    elif a['mode'] == 'reference_event':
        indices = np.array([a['event']['frame']])
    frame = c['reference_frame']; basis = rotation(frame['quaternion_wxyz'])
    kind = c['type']
    if kind == 'gripper_interval':
        return grip[indices]
    if kind == 'axis_angle_interval':
        axis = rotation(pose[indices, 3:]) @ np.asarray(p['local_axis'])
        target = basis @ np.asarray(p['reference_axis'])
        return np.arctan2(np.linalg.norm(np.cross(axis, target), axis=-1), (axis*target).sum(-1))
    if kind == 'angular_displacement_interval':
        r = rotation(pose[a['start'], 3:]).T @ rotation(pose[indices, 3:])
        skew = np.stack((r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1]), -1)
        return np.arctan2(np.linalg.norm(skew, axis=-1)*.5, (np.trace(r, axis1=-2, axis2=-1)-1)*.5)
    origin = pose[a['start'], :3] if kind == 'axis_displacement_interval' else np.asarray(frame['origin_m'])
    delta = pose[indices, :3]-origin
    if kind == 'point_distance_interval':
        return np.linalg.norm(delta, axis=-1)
    axis = basis @ np.asarray(p['axis']); axial = delta @ axis
    return np.linalg.norm(delta-axial[:, None]*axis, axis=-1) if kind == 'radial_distance_interval' else axial


def validate_reference_bounds(bundle, handle):
    """Accepted labels in this initial workflow must admit the source demonstration."""
    report = []
    for c in bundle['constraints']:
        if c['evidence']['status'] != 'accepted':
            continue
        values = reference_values(handle, c)
        lo, hi = c['parameters']['bounds']
        violation = np.maximum(lo-values, 0)+np.maximum(values-hi, 0)
        if not np.isfinite(values).all() or violation.max() > 1e-6:
            raise ValueError(f'Accepted constraint {c["id"]} excludes its reference trajectory')
        report.append({'id': c['id'], 'samples': len(values), 'min': float(values.min()), 'max': float(values.max())})
    return report


def export_review(episode, output):
    import h5py
    episode, output = Path(episode), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with h5py.File(episode, 'r') as h:
        bundle = read_embedded(h)
        if bundle is None or bundle['provenance'].get('native_success') is not True:
            raise ValueError('Review requires a successful clean demonstration with embedded provenance')
        length = len(h['endpose/left_endpose'])
        if length < 2:
            raise ValueError('At least two recorded frames required')
        events, frames = [], set(np.linspace(0, length-1, 10, dtype=int).tolist())
        columns = [np.arange(length)]
        header = ['frame']
        for arm in ('left', 'right'):
            pose = np.asarray(h[f'endpose/{arm}_endpose'])
            rotation(pose[:, 3:])
            grip = np.asarray(h[f'endpose/{arm}_gripper']).reshape(-1)
            if pose.shape != (length, 7) or grip.shape != (length,) or not np.isfinite(pose).all() or not np.isfinite(grip).all():
                raise ValueError('Invalid measured trajectory')
            columns.extend([*pose.T, grip])
            header.extend(f'{arm}_{x}' for x in ('x','y','z','qw','qx','qy','qz','gripper'))
            for direction, sign in (('decreasing', -1), ('increasing', 1)):
                for index in (np.flatnonzero((sign*(grip[:-1]-.5) < 0) & (sign*(grip[1:]-.5) >= 0))+1):
                    events.append({'subject': arm+'_eef', 'frame': int(index), 'direction': direction, 'threshold': .5})
                    frames.update(max(0, min(length-1, int(index)+offset)) for offset in (-2, 0, 3))
        with (output/'trajectory.csv').open('w', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(header); writer.writerows(zip(*columns))
        previews = []
        (output/'images').mkdir()
        for index in sorted(frames):
            for camera in ('head_camera', 'left_camera', 'right_camera'):
                raw = h[f'observation/{camera}/rgb'][index]
                # Native cv2.imencode receives RGB as BGR; restore RGB for PIL.
                pixels = np.asarray(Image.open(io.BytesIO(np.asarray(raw).tobytes())).convert('RGB'))[..., ::-1]
                image = Image.fromarray(pixels); image.thumbnail((384, 384))
                name = f'images/{index:06d}_{camera}.jpg'
                image.save(output/name, quality=82)
                previews.append({'frame': index, 'camera': camera, 'path': name})
        metadata = {'source_sha256': digest(episode), 'recording_id': bundle['provenance']['recording_id'],
                    'frame_count': length, 'instruction': str(h.attrs.get('instruction', '')),
                    'trajectory_file': 'trajectory.csv', 'gripper_events': events, 'previews': previews,
                    'bundle_template': bundle, 'annotation_policy': {
                        'position_margin_m': .005, 'angle_margin_rad': .0872664626, 'gripper_margin': .02,
                        'meaning': 'Operator defaults for reference envelopes, not experimentally verified physical tolerance. '
                                   'Do not infer grasp/contact/support boundaries; mark unsupported geometry proposed or unknown.'}}
        if 'robotwin_expert_provenance' in h.attrs:
            from services.robotwin.programmatic_annotation import trajectory_data
            _, arms, signature = trajectory_data(h)
            metadata['expert_provenance'] = json.loads(h.attrs['robotwin_expert_provenance'])
            metadata['task'] = metadata['expert_provenance']['task']
            metadata['event_signature'] = signature
            metadata['active_arms'] = [arm for arm, data in arms.items() if data['active']]
        (output/'review.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    return metadata


def embed_review(episode, proposal, *, agent_run):
    episode = Path(episode).resolve()
    with episode.with_name('.'+episode.name+'.annotation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _embed_review(episode, proposal, agent_run=agent_run)


def _embed_review(episode, proposal, *, agent_run):
    import h5py
    episode = Path(episode)
    if set(proposal) != {'source_sha256', 'constraints'}:
        raise ValueError('Proposal requires exactly source_sha256 and constraints')
    if digest(episode) != proposal['source_sha256']:
        raise ValueError('Stale annotation: source HDF5 changed')
    with h5py.File(episode, 'r') as h:
        bundle = read_embedded(h)
        if bundle is None or bundle['provenance'].get('native_success') is not True:
            raise ValueError('Only successful clean training recordings may be annotated')
        bundle = copy.deepcopy(bundle); bundle['constraints'] = proposal['constraints']
        validate_bundle(bundle, len(h['endpose/left_endpose']))
        validate_reference_events(bundle, h)
        audit = validate_reference_bounds(bundle, h)
    temporary = episode.with_name('.'+episode.name+'.annotating-'+uuid.uuid4().hex)
    try:
        shutil.copy2(episode, temporary)
        with h5py.File(temporary, 'r+') as h:
            history = json.loads(h['trajectory_constraints/history'][()])
            revision = int(h['trajectory_constraints'].attrs['revision'])+1
            history.append({'revision': revision, 'agent_run': str(agent_run),
                            'source_sha256': proposal['source_sha256'], 'constraints': proposal['constraints'],
                            'reference_audit': audit})
            h[HDF5_KEY][()] = canonical_json(bundle)
            h['trajectory_constraints/history'][()] = json.dumps(history, ensure_ascii=False)
            h['trajectory_constraints'].attrs['revision'] = revision
            read_embedded(h)
        if digest(episode) != proposal['source_sha256']:
            raise ValueError('Concurrent annotation changed the source')
        os.replace(temporary, episode)
    finally:
        temporary.unlink(missing_ok=True)
    return {'episode': str(episode), 'sha256': digest(episode), 'revision': revision,
            'accepted': len(audit), 'constraints': len(proposal['constraints']), 'reference_audit': audit}
