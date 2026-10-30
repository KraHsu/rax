"""Reconstruct full policy commands from immutable raw collector evidence."""
import json
import math
from pathlib import Path

import numpy as np

from services.robotwin.acceptance import file_hash, json_hash


def read(path):
    return json.loads(Path(path).read_text())


def quaternion_xyzw_matrix(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError('Invalid converted model quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def verify_raw_conversion(raw_rows, trajectory):
    """Independent algebra, no imported or generated collector code execution.

    The declared legacy checkpoint roundtrip writes the helper's numeric XYZW
    tuple directly into native WXYZ fields. Verify that exact existing adapter,
    rather than silently correcting the checkpoint's encoding during audit.
    """
    n = len(trajectory['left']['pose'])
    if len(raw_rows) != n-1 or [r['frame'] for r in raw_rows] != list(range(1, n)):
        raise ValueError('Raw model outputs must cover the original complete command horizon')
    clipped = 0
    for record in raw_rows:
        raw = np.asarray(record['eef20'], dtype=float)
        if raw.shape != (20,) or not np.isfinite(raw).all():
            raise ValueError('Invalid raw EEF20 policy output')
        this_clip = False
        for arm, offset in [('left', 0), ('right', 10)]:
            pose = np.asarray(trajectory[arm]['pose'][record['frame']], dtype=float)
            grip = trajectory[arm]['gripper'][record['frame']]
            if not np.array_equal(pose[:3], raw[offset:offset+3].astype(np.float32).astype(float)):
                raise ValueError('Model command position differs from raw output')
            a, b = raw[offset+3:offset+6], raw[offset+6:offset+9]
            if np.linalg.norm(a) < 1e-8:
                raise ValueError('Degenerate raw rotation column')
            a = a/np.linalg.norm(a); b = b-np.dot(a, b)*a
            if np.linalg.norm(b) < 1e-8:
                raise ValueError('Degenerate raw rotation columns')
            b = b/np.linalg.norm(b)
            expected = np.stack((a, b, np.cross(a, b)), axis=1)
            if not np.allclose(quaternion_xyzw_matrix(pose[3:]), expected, atol=1e-6, rtol=0):
                raise ValueError('Model command orientation differs from declared raw conversion')
            raw_grip = float(np.float32(raw[offset+9]))
            expected_grip = float(np.clip(raw_grip, 0, 1))
            this_clip |= raw_grip != expected_grip
            if grip != expected_grip:
                raise ValueError('Model command gripper differs from declared native clipping')
        if record.get('gripper_clipped') is not this_clip:
            raise ValueError('Raw clipping disclosure differs from actual conversion')
        clipped += this_clip
    return dict(gripper_clip_0_1_frames=clipped, time_resampling=False, expert_action_blend=False)


def verify_model_evidence(inventory_path, rows, release):
    inventory = read(inventory_path)
    if set(inventory) != {'version', 'runs'} or inventory['version'] != 1 or not isinstance(inventory['runs'], list):
        raise ValueError('Explicit raw policy evidence inventory required')
    runs, hashes, proofs = {}, {}, []
    def digest(path):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = file_hash(path)
        return hashes[path]
    for spec in inventory['runs']:
        if set(spec) != {'root', 'checkpoint_path', 'checkpoint_receipt_path', 'server_config_path',
                         'server_receipt_path', 'collector_path'}:
            raise ValueError('Raw model evidence requires source, checkpoint and collector artifact paths')
        root = Path(spec['root']).resolve()
        run = read(root/'source_run.json'); run_hash = digest(root/'source_run.json')
        if run_hash in runs:
            raise ValueError('Duplicate raw model run')
        for field, name in [('checkpoint_sha256', 'checkpoint_path'),
                            ('checkpoint_receipt_sha256', 'checkpoint_receipt_path'),
                            ('server_config_sha256', 'server_config_path'),
                            ('server_receipt_sha256', 'server_receipt_path'), ('collector_sha256', 'collector_path')]:
            if run.get(field) != digest(spec[name]):
                raise ValueError('Model source checkpoint/config/collector artifact changed')
        if (run.get('source_kind') != 'model_closed_loop'
                or run.get('rotation_encoding') != 'checkpoint_legacy_reader_client_roundtrip_native_wxyz'
                or type(run.get('created_unix')) not in (float, int)
                or not math.isfinite(run['created_unix'])
                or run['created_unix'] < release['frozen_unix']):
            raise ValueError('Final raw policy run must postdate the frozen release and declare its action adapter')
        config, result = read(root/'configuration.json'), read(root/'result.json')
        if any(run.get(k) != v for k, v in config.items()) or result.get('complete') is not True or result.get('error') is not None:
            raise ValueError('Raw policy collection configuration changed or rollout is incomplete')
        candidates = read(root/'model_candidates.json')
        raw, observations = read(root/'raw_policy_actions.json'), read(root/'observation_hashes.json')
        if [r.get('frame') for r in observations] != list(range(1, run['horizon']+1)):
            raise ValueError('Full policy rollout lacks the original per-step observation receipts')
        for obs in observations:
            state = np.asarray(obs.get('state'), dtype=float)
            camera_hashes = obs.get('camera_sha256', {})
            if (state.shape != (20,) or not np.isfinite(state).all() or not camera_hashes
                    or any(not isinstance(v, str) or len(v) != 64 or set(v)-set('0123456789abcdef') for v in camera_hashes.values())):
                raise ValueError('Invalid original policy observation receipt')
        if any(candidates.get(k) != run.get(k) for k in ('task', 'scene_seed', 'scene_sha256', 'alignment', 'action_contract')):
            raise ValueError('Original model candidates differ from their native source run')
        originals = {}
        for candidate in candidates['candidates']:
            p = candidate['model_provenance']; trajectory = candidate['trajectory']
            if (p.get('source_run_sha256') != run_hash or p.get('checkpoint_sha256') != run['checkpoint_sha256']
                    or p.get('raw_policy_actions_sha256') != digest(root/'raw_policy_actions.json')
                    or p.get('observation_hashes_sha256') != digest(root/'observation_hashes.json')):
                raise ValueError('Model provenance does not bind original raw outputs and observations')
            transform = verify_raw_conversion(raw, trajectory)
            if p.get('applied_action_transform') != transform or len(trajectory['left']['pose']) != run['horizon']+1:
                raise ValueError('Declared model action transform or horizon differs from raw commands')
            originals[json_hash(trajectory)] = candidate
        runs[run_hash] = (run, originals)
        proofs.append(dict(root=str(root), source_run_sha256=run_hash,
                           files={name: digest(root/name) for name in ('configuration.json', 'result.json',
                               'model_candidates.json', 'raw_policy_actions.json', 'observation_hashes.json')}))
    used = set()
    for row in rows:
        if row['source'] != 'model' or row.get('kind') == 'baseline':
            continue
        p = row['model_provenance']; source = p['source_run_sha256']; used.add(source)
        if source not in runs:
            raise ValueError('Actual model holdout has no auditable original raw policy source')
        run, originals = runs[source]
        if (row['trajectory_sha256'] not in originals
                or p.get('unmodified_model_trajectory_sha256') != row['trajectory_sha256']
                or any(row.get(k) != run.get(k) for k in ('task', 'scene_seed', 'scene_sha256'))):
            raise ValueError('Actual model holdout command is not the original full policy trajectory')
        original = originals[row['trajectory_sha256']]['model_provenance']
        if any(p.get(k) != v for k, v in original.items()):
            raise ValueError('Candidate changed its original model provenance')
    if used != set(runs):
        raise ValueError('Raw model proof inventory must cover exactly the tested model runs')
    return dict(version=1, inventory_sha256=file_hash(inventory_path), runs=proofs)
