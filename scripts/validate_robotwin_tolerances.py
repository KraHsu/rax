#!/usr/bin/env python3
"""Empirical clean-scene tolerance scans using a frozen native expert program.

No training HDF5 is created or edited. Physics snapshots avoid renderer startup
per intervention; three zero-perturbation replays must pass before scanning.
"""
import argparse
import copy
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.robotwin.catalog import EpisodeManifest, source_root
from services.robotwin.runtime import initialize_runtime, native_config, source_commit
from services.robotwin.tolerance import bracket, features, initial_cases, perturb_program, realization


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path.write_text(json.dumps(value, default=plain, ensure_ascii=False, indent=2, allow_nan=False))


def serialize(groups):
    return [{'arm':str(group[0]), 'actions':[{
        'arm_tag':str(a.arm_tag), 'action':a.action,
        'target_pose':copy.deepcopy(a.target_pose),
        'target_gripper_pos':a.target_gripper_pos,
        'args':copy.deepcopy(a.args)} for a in group[1]]}
        for group in groups if group is not None]


def deserialize(groups):
    from envs.utils.action import Action, ArmTag
    # Native Action('gripper') omits assigning target_gripper_pos; 'close'
    # accepts an explicit fraction and produces the same gripper action.
    return [(ArmTag(g['arm']), [Action(a['arm_tag'], 'move' if a['action']=='move' else 'close', target_pose=a['target_pose'],
                target_gripper_pos=a['target_gripper_pos'], **a['args']) for a in g['actions']]) for g in groups]


class Experiment:
    def __init__(self, native, output, seed):
        self.native, self.output, self.seed = native, output, seed
        self.arm = 'right' if native.qpose_tag else 'left'
        self.program, self.samples, self.results = [], [], []
        self.stage = -1
        self.original_move = native.move
        self.physics = native.scene.physx_system.pack()
        self.drives = [(j, np.array(j.get_drive_target()).copy(), np.array(j.get_drive_velocity_target()).copy())
                       for a in native.scene.get_all_articulations() for j in a.get_active_joints()]
        self.openings = (native.robot.left_gripper_val, native.robot.right_gripper_val)
        self.initial = self.sample(store=False)
        # Native calls this at exactly its recording cadence. Only measurements
        # are stored; no images, cache files, or training recordings are written.
        native._take_picture = lambda *args, **kwargs: self.sample()

    def sample(self, store=True):
        n = self.native; r = n.robot
        value = {'stage':self.stage, 'left_pose':list(r.get_left_ee_pose()),
                 'right_pose':list(r.get_right_ee_pose()),
                 'left_opening':float(r.get_left_gripper_val()),
                 'right_opening':float(r.get_right_gripper_val()),
                 'bottle_functional_pose':np.asarray(n.bottle.get_functional_point(0)).tolist()}
        if store: self.samples.append(value)
        return value

    def capture(self):
        def move(*groups, **kwargs):
            self.program.append(serialize(groups)); self.stage = len(self.program)-1
            return self.original_move(*groups, **kwargs)
        self.native.move = move
        try:
            self.native.play_once()
        finally:
            self.native.move = self.original_move
        if not self.native.plan_success or not self.native.check_success():
            raise RuntimeError('Native expert baseline did not succeed')
        write(self.output/'expert_program.json', self.program)
        write(self.output/'expert_trace.json', self.samples)
        self.reference = features(self.samples, self.arm)
        write(self.output/'reference.json', self.reference)

    def load_program(self, source):
        """Fresh-scene confirmation of an existing frozen program, not relearning."""
        source=Path(source)
        previous=json.loads((source/'run.json').read_text())
        initial=json.loads((source/'expert_trace.json').read_text())[0]
        if previous['seed']!=self.seed or previous['model_id']!=int(self.native.model_id):
            raise RuntimeError('Fresh scene identity differs from the source program')
        for key in ('left_pose','right_pose','bottle_functional_pose'):
            if not np.allclose(self.initial[key],initial[key],atol=1e-5,rtol=0):
                raise RuntimeError(f'Fresh initial geometry differs at {key}')
        self.program=json.loads((source/'expert_program.json').read_text())
        self.reference=json.loads((source/'reference.json').read_text())
        write(self.output/'expert_program.json',self.program)
        write(self.output/'reference.json',self.reference)
        write(self.output/'fresh_initial_state.json',self.initial)

    def reset(self):
        import torch
        n = self.native
        n.scene.physx_system.unpack(self.physics)
        for j, position, velocity in self.drives:
            j.set_drive_target(position); j.set_drive_velocity_target(velocity)
        n.robot.left_gripper_val, n.robot.right_gripper_val = self.openings
        n.plan_success = True; n.eval_success = False; n.take_action_cnt = 0
        n.left_joint_path = []; n.right_joint_path = []; n.left_cnt = n.right_cnt = 0
        n.FRAME_IDX = 0; n.now_obs = {}; n.run_steps = 0
        np.random.seed(self.seed); torch.manual_seed(self.seed); random.seed(self.seed)
        self.samples = []; self.stage = -1
        observed = self.sample(store=False)
        for key in ('left_pose', 'right_pose', 'bottle_functional_pose'):
            if not np.allclose(observed[key], self.initial[key], atol=1e-6, rtol=0):
                raise RuntimeError(f'Physics checkpoint did not restore {key}')

    def execute_program(self, program):
        for self.stage, groups in enumerate(program):
            self.original_move(*deserialize(groups))
            if not self.native.plan_success: break

    def trial_details(self):
        return {}

    def trial(self, delta, phase):
        started = time.monotonic(); self.reset()
        record = {'trial':len(self.results), 'phase':phase, 'delta':delta, 'seed':self.seed}
        try:
            program = perturb_program(self.program, delta)
            self.execute_program(program)
            self.sample()
            observed = features(self.samples,self.arm)
            end_success = bool(self.native.check_success())
            # Additional stability check; preserve the original endpoint feature
            # and report the native end verdict separately from this hold.
            for _ in range(125): self.native.scene.step()
            hold = self.sample(store=False)
            held_success = bool(self.native.check_success())
            record.update(outcome=('planner_failure' if not self.native.plan_success else
                          'success' if end_success and held_success else 'task_failure'),
                          plan_success=bool(self.native.plan_success),
                          native_success=end_success, hold_success=held_success,
                          hold_seconds=125*self.native.scene.get_timestep(), hold_state=hold,
                          features=observed, **realization(self.reference,observed,delta),
                          **self.trial_details())
        except Exception as exc:
            record.update(outcome='execution_error', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['seconds'] = time.monotonic()-started
            self.results.append(record)
            write(self.output/f'trace_{record["trial"]:04d}.json', self.samples)
            with (self.output/'results.jsonl').open('a') as f:
                f.write(json.dumps(record, default=plain, allow_nan=False)+'\n')
            print(json.dumps(record, default=plain, allow_nan=False), flush=True)
        return record

    def baselines(self):
        for _ in range(3):
            r = self.trial({}, 'baseline')
            if r['outcome'] != 'success': raise RuntimeError('Unperturbed replay failed; scan aborted')
            for key in ('closure_xyz_m', 'terminal_xyz_m'):
                if r['features'][key] is None or np.linalg.norm(np.array(r['features'][key])-self.reference[key]) > .002:
                    raise RuntimeError(f'Unperturbed replay drift exceeds 2 mm at {key}; scan aborted')

    def scan(self):
        for delta in initial_cases(): self.trial(delta, 'coarse')
        axes = ('grasp_x','grasp_y','grasp_z','opening','terminal_z')
        for axis in axes:
            for sign in ((1,) if axis == 'opening' else (-1,1)):
                extra = (1.,) if axis=='opening' else (.16,.24) if axis=='terminal_z' else (.08,.12)
                for value in extra:
                    if bracket(self.results,axis,sign)['nonpass'] is not None:break
                    self.trial({axis:sign*value},'expand')
        for axis in axes:
            for sign in ((1,) if axis == 'opening' else (-1,1)):
                for _ in range(3):
                    b = bracket(self.results,axis,sign)
                    resolution = .005 if axis=='opening' else .0025
                    if b['nonpass'] is None or abs(b['nonpass']-b['pass']) <= resolution: break
                    self.trial({axis:(b['pass']+b['nonpass'])/2}, 'refine')
        bounds = {axis:{str(sign):bracket(self.results,axis,sign)
                        for sign in ((1,) if axis=='opening' else (-1,1))} for axis in axes}
        write(self.output/'one_axis_brackets.json',bounds)
        # Even independently successful one-axis offsets need joint validation.
        choices = [[bounds[a]['-1']['pass']*.5,bounds[a]['1']['pass']*.5] for a in axes[:3]]
        for xyz in itertools.product(*choices):
            self.trial(dict(zip(axes[:3],xyz)), 'inner_position_corner')
        opening = bounds['opening']['1']['pass']*.5
        for sign in (-1,1):
            self.trial({'opening':opening,'terminal_z':bounds['terminal_z'][str(sign)]['pass']*.5}, 'opening_height_pair')
        # Repeat bracket endpoints and a zero intervention after the sweep.
        for axis in axes:
            for sign in ((1,) if axis=='opening' else (-1,1)):
                b=bounds[axis][str(sign)]
                if b['pass'] != 0:self.trial({axis:b['pass']},'boundary_repeat')
                if b['nonpass'] is not None:self.trial({axis:b['nonpass']},'boundary_repeat')
        r=self.trial({},'baseline_after_scan')
        if r['outcome']!='success':raise RuntimeError('End-of-scan baseline failed')
        write(self.output/'replicated_brackets.json', {axis:{str(sign):bracket(self.results,axis,sign)
              for sign in ((1,) if axis=='opening' else (-1,1))} for axis in axes})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline-only',action='store_true')
    p.add_argument('--cases-json',type=Path,help='Explicit intervention list for independent confirmation')
    p.add_argument('--program-from',type=Path,help='Replay an existing frozen program in a newly initialized matching scene')
    args=p.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    cases=json.loads(args.cases_json.resolve().read_text()) if args.cases_json else None
    program_from=args.program_from.resolve() if args.program_from else None
    manifest=EpisodeManifest(args.manifest.resolve(),'adjust_bottle'); entry=manifest.entry(0,args.index)
    if source_commit()!=manifest.source_commit:raise ValueError('Native source commit differs from manifest')
    metadata={'status':'initializing','seed':entry['seed'],'collection':'clean','manifest_sha256':manifest.sha256,
              'native_commit':manifest.source_commit,'started_unix':time.time(),'interventions':'frozen_expert_targets',
              'scope':'simulation-local empirical evidence; not a proof of a continuous feasible set',
              'hold_seconds':.5, 'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'program_from':str(program_from) if program_from else None,
              'cases_sha256':hashlib.sha256(args.cases_json.resolve().read_bytes()).hexdigest() if args.cases_json else None,
              'tolerance_module_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'services/robotwin/tolerance.py').read_bytes()).hexdigest(),
              'native_source_sha256':{f:hashlib.sha256((source_root()/f).read_bytes()).hexdigest()
                for f in ('envs/adjust_bottle.py','envs/_base_task.py','envs/robot/robot.py')}}
    write(output/'run.json',metadata); print(json.dumps(metadata),flush=True)
    native=None
    try:
        initialize_runtime()
        native=importlib.import_module('envs.adjust_bottle').adjust_bottle()
        native.setup_demo(now_ep_num=args.index,seed=entry['seed'],is_test=False,**native_config('adjust_bottle',0))
        native.set_instruction(entry['instruction'])
        experiment=Experiment(native,output,entry['seed'])
        if program_from:experiment.load_program(program_from)
        else:experiment.capture()
        experiment.baselines()
        metadata.update(status='scanning',arm=experiment.arm,model_id=int(native.model_id));write(output/'run.json',metadata)
        if cases is not None:
            for delta in cases:experiment.trial(delta,'confirmation')
            if experiment.trial({},'baseline_after_confirmation')['outcome']!='success':
                raise RuntimeError('End-of-confirmation baseline failed')
        elif not args.baseline_only:experiment.scan()
        metadata.update(status='completed',trials=len(experiment.results))
    except BaseException as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if native is not None:native.close_env()
        metadata['finished_unix']=time.time();write(output/'run.json',metadata)


if __name__=='__main__':main()
