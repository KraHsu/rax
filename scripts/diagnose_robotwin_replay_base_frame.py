#!/usr/bin/env python3
"""Private SE(3) replay probe; never publishes acceptance or edits native source."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np


def install_private_base_frame_adapter(native):
    """Correct only this planner instance's omitted fixed arm-base rotation."""
    import transforms3d as t3d
    import yaml
    robot = native.robot
    if robot.communication_flag or robot.planner_backend != 'curobo':
        raise ValueError('Probe supports only in-process cuRobo planners')
    prepared, proof = [], {}
    for arm in ('left', 'right'):
        planner = getattr(robot, arm+'_planner')
        if getattr(planner, '_rax_private_base_frame', False):
            raise ValueError('Base-frame adapter already installed')
        config_path = Path(planner.yml_path)
        cfg = yaml.safe_load(config_path.read_text())['robot_cfg']['kinematics']
        planner_urdf = Path(os.path.expandvars(cfg['urdf_path'])).resolve()
        physical_urdf = Path(getattr(robot, arm+'_urdf_path')).resolve()
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        if digest(planner_urdf) != digest(physical_urdf):
            raise ValueError('Planner and physics URDF differ')
        urdf = ET.fromstring(physical_urdf.read_text())
        joints = urdf.findall('joint')
        roots = {n.get('name') for n in urdf.findall('link')} - {j.find('child').get('link') for j in joints}
        matches = [j for j in joints if j.find('child').get('link') == cfg['base_link']]
        if len(roots) != 1 or len(matches) != 1:
            raise ValueError('Unsupported ambiguous URDF arm-base root chain')
        joint = matches[0]
        if joint.get('type') != 'fixed' or joint.find('parent').get('link') not in roots:
            raise ValueError('Arm base must be a direct fixed child of the URDF root')
        origin = joint.find('origin')
        if origin is None:
            raise ValueError('Fixed arm base lacks an explicit transform')
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(np.r_[xyz, rpy]).all():
            raise ValueError('Invalid fixed arm-base transform')
        bias = np.asarray(planner.frame_bias, dtype=float)
        if bias.shape != (3,) or not np.allclose(xyz, -bias, rtol=0, atol=1e-8):
            raise ValueError('Native frame_bias does not match the fixed URDF translation')
        rotation = t3d.euler.euler2mat(*rpy, axes='sxyz')
        original = planner._trans_from_world_to_base
        def corrected(base_pose, target_pose, _original=original, _rotation=rotation, _bias=bias.copy()):
            p_root, q_root = _original(base_pose, target_pose)
            # Native plan_path subsequently adds frame_bias. Return the value
            # before that addition, so it is applied exactly once.
            position = _rotation.T @ (np.asarray(p_root)+_bias)-_bias
            orientation = t3d.quaternions.mat2quat(_rotation.T @ t3d.quaternions.quat2mat(q_root))
            return position, orientation
        prepared.append((planner, corrected))
        proof[arm] = dict(urdf_path=str(physical_urdf), urdf_sha256=digest(physical_urdf),
                         planner_yaml=str(config_path), planner_yaml_sha256=digest(config_path),
                         root_link=next(iter(roots)), arm_base=cfg['base_link'], fixed_joint=joint.get('name'),
                         xyz=xyz.tolist(), rpy=rpy.tolist(), frame_bias=bias.tolist(),
                         rotation=rotation.tolist(), quaternion_order='wxyz')
    for planner, corrected in prepared:
        planner._trans_from_world_to_base = corrected
        planner._rax_private_base_frame = True
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Diagnostic interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupt)
    root, output = args.root.resolve(), args.output.resolve()
    if root/'replay_diagnostics' not in output.parents:
        raise ValueError('Output must remain under replay_diagnostics')
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'code/rax'))
    from services.robotwin.acceptance_replay import replay, tracking_report
    from services.robotwin.collection import atomic_json, initial_scene, install_planner_cache, scene_fingerprint
    from services.robotwin.runtime import initialize_runtime, native_config, source_commit
    initialize_runtime()
    install_planner_cache()
    for task in ('adjust_bottle', 'beat_block_hammer', 'lift_pot'):
        directory = output/task
        directory.mkdir()
        manifests = list((root/'acceptance'/task).glob('*/experiment_manifest.json'))
        if len(manifests) != 1:
            raise ValueError('Expected exactly one sealed task manifest')
        manifest = json.loads(manifests[0].read_text())
        entry = manifest['reference_artifacts'][0]
        path = manifests[0].parent/entry['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Frozen reference changed')
        reference = json.loads(path.read_text())
        report = dict(kind='diagnostic_only_not_acceptance', pid=os.getpid(), status='initializing',
                      task=task, seed=entry['seed'], started_unix=time.time(), source_commit=source_commit(),
                      reference_path=str(path), reference_sha256=entry['sha256'])
        atomic_json(directory/'result.json', report)
        native = getattr(importlib.import_module('envs.'+task), task)()
        kwargs = native_config(task, 0)
        kwargs.update(need_plan=True, save_data=False, render_freq=0, eval_video_log=False,
                      eval_video_save_dir=None, save_path=str(directory/'scratch'))
        random.seed(entry['seed'])
        try:
            native.setup_demo(now_ep_num=0, seed=entry['seed'], is_test=False, **kwargs)
            if scene_fingerprint(initial_scene(native)) != reference['provenance']['scene_sha256']:
                raise ValueError('Diagnostic scene differs from sealed reference')
            report['adapter'] = install_private_base_frame_adapter(native)
            report['status'] = 'replaying'
            atomic_json(directory/'result.json', report)
            raw = replay(native, reference['trajectory'])
            actual = raw.pop('actual')
            atomic_json(directory/'trajectory.json', dict(commanded=reference['trajectory'], actual=actual))
            atomic_json(directory/'outcome.json', raw)
            report['outcome'] = raw
            report['tracking'] = tracking_report(actual, reference['trajectory'])
            report['passed'] = all([raw['native_success'], raw['planner_success'], raw['execution_complete'],
                                    report['tracking']['realized']])
            report.update(status='complete', finished_unix=time.time())
            atomic_json(directory/'result.json', report)
            print(json.dumps(dict(task=task, passed=report['passed'], tracking=report['tracking'])), flush=True)
            if not report['passed']:
                raise RuntimeError('Corrected baseline did not pass; remaining probes withheld')
        except BaseException as exc:
            report['error'] = f'{type(exc).__name__}: {exc}'
            if report['status'] != 'complete':
                report['status'] = 'failed'
            atomic_json(directory/'result.json', report)
            raise
        finally:
            native.close_env()


if __name__ == '__main__':
    main()
