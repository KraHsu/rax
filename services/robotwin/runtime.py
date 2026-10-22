"""Native RoboTwin configuration; imported in the simulator interpreter only."""
import importlib
import os
from pathlib import Path
import subprocess
import sys

import yaml

from services.robotwin.catalog import COLLECTIONS, check_task, source_root, step_limit


def source_commit():
    return subprocess.check_output(['git', '-C', str(source_root()), 'rev-parse', 'HEAD'], text=True).strip()


def native_config(task, collection):
    check_task(task)
    root = source_root()
    config_root = root / 'task_config'
    args = yaml.safe_load((config_root / f'{COLLECTIONS[collection]}.yml').read_text())
    embodiments = yaml.safe_load((config_root / '_embodiment_config.yml').read_text())
    types = args['embodiment']
    if len(types) not in (1, 3):
        raise ValueError('Native embodiment must contain one or three entries')
    for arm, kind in [('left', types[0]), ('right', types[0] if len(types) == 1 else types[1])]:
        path = Path(embodiments[kind]['file_path'])
        if not path.is_absolute():
            path = root / path
        args[f'{arm}_robot_file'] = str(path)
        args[f'{arm}_embodiment_config'] = yaml.safe_load((path / 'config.yml').read_text())
    args['dual_arm_embodied'] = len(types) == 1
    if len(types) == 3:
        args['embodiment_dis'] = types[2]
    cameras = yaml.safe_load((config_root / '_camera_config.yml').read_text())
    head = cameras[args['camera']['head_camera_type']]
    args.update(head_camera_h=head['h'], head_camera_w=head['w'], task_name=task,
                task_config=COLLECTIONS[collection], eval_mode=collection == 1,
                render_freq=0, save_data=False, eval_video_save_dir=None, eval_video_log=False)
    args['data_type'] = {k: k in {'rgb', 'qpos', 'endpose'} for k in args.get('data_type', {})}
    args['data_type'].update(rgb=True, qpos=True, endpose=True)
    return args


def initialize_runtime():
    root = source_root()
    # Native assets/configs are resolved relative to the RoboTwin checkout.
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / 'script'))
    import torch
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    # cuRobo must establish its CUDA context before SAPIEN scene creation.
    importlib.import_module('curobo.wrap.reacher.motion_gen')


def make_environment(task, collection, entry, *, episode_index):
    check_task(task)
    native = getattr(importlib.import_module(f'envs.{task}'), task)()
    try:
        native.setup_demo(now_ep_num=episode_index, seed=entry['seed'], is_test=collection == 1,
                          **native_config(task, collection))
        native.step_lim = step_limit(task)  # clean collection also has a command budget
        native.set_instruction(instruction=entry['instruction'])
        return native
    except BaseException:
        native.close_env()
        raise
