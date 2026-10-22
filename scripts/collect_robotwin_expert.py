#!/usr/bin/env python3
"""Record clean expert demonstrations with RoboTwin's native plan/replay writer."""
import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.robotwin.catalog import EpisodeManifest
from services.robotwin.runtime import initialize_runtime, native_config, source_commit


def publish_scene_info(output, index, info):
    """Keep the native episode_<index> metadata layout used by OpenWAM."""
    output = Path(output)
    with (output/'.scene_info.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output/'scene_info.json'
        data = json.loads(path.read_text()) if path.exists() else {}
        data[f'episode_{index}'] = info
        temporary = output/('.scene_info.'+uuid.uuid4().hex+'.json')
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def collect(task, manifest, indices, output):
    """Use the same native two-pass methods as script/collect_data.py.

    Only clean manifest entries are selectable. Each episode gets a private
    staging directory, so a failed replay never appears in the training data.
    """
    import h5py
    from services.robotwin.constraint_schema import HDF5_KEY, canonical_json
    if source_commit() != manifest.source_commit:
        raise ValueError('RoboTwin source differs from the episode manifest')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for index in indices:
        entry = manifest.entry(0, index)
        stage = output/'native'/f'episode{index}'
        stage.mkdir(parents=True, exist_ok=False)
        target = output/'data'/f'episode{index}.hdf5'
        if target.exists():
            raise FileExistsError(target)
        args = native_config(task, 0)
        args.update(save_path=str(stage), need_plan=True, save_data=False,
                    eval_video_log=False, render_freq=0)
        native = getattr(importlib.import_module(f'envs.{task}'), task)()
        print(json.dumps({'phase': 'expert_plan', 'index': index, 'seed': entry['seed']}), flush=True)
        try:
            native.setup_demo(now_ep_num=index, seed=entry['seed'], is_test=False, **args)
            native.set_instruction(entry['instruction'])
            native.play_once()
            if not (native.plan_success and native.check_success()):
                raise RuntimeError(f'Expert planning failed for clean seed {entry["seed"]}')
            native.save_traj_data(index)
        finally:
            native.close_env()
        args.update(need_plan=False, save_data=True)
        print(json.dumps({'phase': 'native_recording_replay', 'index': index}), flush=True)
        try:
            native.setup_demo(now_ep_num=index, seed=entry['seed'], is_test=False, **args)
            native.set_instruction(entry['instruction'])
            paths = native.load_tran_data(index)
            native.set_path_lst(dict(args, **paths))
            info = native.play_once()
            # Check while the native scene is alive; close_env destroys it.
            success = bool(native.check_success())
            if not success:
                raise RuntimeError(f'Expert recording replay failed for clean seed {entry["seed"]}')
        finally:
            native.close_env()
        native.merge_pkl_to_hdf5_video()
        native.remove_data_cache()
        source = stage/'data'/f'episode{index}.hdf5'
        recording_id = f'{manifest.sha256[:16]}-clean-{index}'
        bundle = {'schema_version': '1.0', 'action_contract': {
            'position': 'absolute', 'rotation': 'absolute', 'quaternion_order': 'wxyz',
            'gripper': 'absolute_open_fraction', 'frame': 'world'}, 'provenance': {
            'split': 'train', 'variant': 'clean_50', 'recording_id': recording_id,
            'source': 'robotwin_native_expert_rax_annotation', 'native_success': True,
            'sampling': f'robotwin_native_save_freq_{args["save_freq"]}'}, 'constraints': []}
        with h5py.File(source, 'r+') as handle:
            handle.attrs['instruction'] = entry['instruction']
            handle.attrs['robotwin_expert_provenance'] = json.dumps({
                'manifest_sha256': manifest.sha256, 'source_commit': manifest.source_commit,
                'seed': entry['seed'], 'collection': 0, 'native_success': success,
                'recording_method': 'native_plan_replay_merge_pkl_to_hdf5_video',
                'save_freq': args['save_freq']})
            handle.create_dataset(HDF5_KEY, data=canonical_json(bundle), dtype=h5py.string_dtype('utf-8'))
            handle.create_dataset('trajectory_constraints/history', data='[]', dtype=h5py.string_dtype('utf-8'))
            handle['trajectory_constraints'].attrs['revision'] = 0
            frames = len(handle['endpose/left_endpose'])
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_scene_info(output, index, info)
        source.rename(target)
        (stage/'scene_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2))
        receipt = {'episode': str(target), 'frames': frames, 'seed': entry['seed'],
                   'native_success': success, 'constraints': 0, 'annotation_status': 'pending'}
        (stage/'receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--indices', type=int, nargs='+', default=[0])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = EpisodeManifest(args.manifest, args.task)
    output = args.output.resolve()
    if len(set(args.indices)) != len(args.indices):
        p.error('Duplicate episode indices')
    for index in args.indices:
        manifest.entry(0, index)
    print(json.dumps({'phase': 'initialize_runtime', 'task': args.task}), flush=True)
    initialize_runtime()
    collect(args.task, manifest, args.indices, output)


if __name__ == '__main__':
    main()
