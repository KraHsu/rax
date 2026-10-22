#!/usr/bin/env python3
"""Matched clean-scene interventions with action, contact and joint diagnostics.

The adaptive arm recomputes the native expert's lift/place continuation after
the SAME perturbed grasp. It tests grasp feasibility with a suitable continuation;
it must not be treated as validation of the frozen full action chunk.
"""
import argparse
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.validate_robotwin_tolerances import Experiment, deserialize, serialize, write
from services.robotwin.catalog import EpisodeManifest
from services.robotwin.runtime import initialize_runtime, native_config, source_commit
from services.robotwin.annotation import rotation
from services.robotwin.tolerance import retarget_position


class DiagnosticExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        self.mode = 'frozen'
        self.action_index = -1
        self.actions = []
        self.grasp_checkpoint = None
        super().__init__(*args, **kwargs)

    def sample(self, store=True):
        value = super().sample(store=False)
        n = self.native; robot = n.robot
        value['action_index'] = self.action_index
        value['bottle_pose'] = list(n.bottle.get_pose().p) + list(n.bottle.get_pose().q)
        articulation = getattr(robot, self.arm+'_entity')
        joints = articulation.get_active_joints()
        qpos = articulation.get_qpos()
        value['joints'] = {j.name: {'qpos':float(qpos[i]), 'target':float(j.get_drive_target()[0])}
                           for i,j in enumerate(joints)}
        primary = getattr(robot,self.arm+'_gripper')[0][0]
        scale = getattr(robot,self.arm+'_gripper_scale')
        value['physical_gripper_fraction'] = float((qpos[joints.index(primary)]-scale[0])/(scale[1]-scale[0]))
        contacts = []
        for c in n.scene.get_contacts():
            names = [b.entity.name for b in c.bodies]
            # Include all nonzero contact impulses: table/object and robot/object
            # are needed to distinguish contact saturation from planning error.
            impulses = [float(np.linalg.norm(p.impulse)) for p in c.points]
            if sum(impulses)>1e-8:
                contacts.append({'bodies':names, 'impulse_Ns':sum(impulses),
                                 'positions':[np.asarray(p.position).tolist() for p in c.points]})
        value['contacts'] = contacts
        if store:self.samples.append(value)
        return value

    def load_program(self, source):
        super().load_program(source)
        trace=json.loads((Path(source)/'expert_trace.json').read_text())
        state=[s for s in trace if s['stage']==1][-1]
        pose=state[self.arm+'_pose']
        self.reference_object_in_eef=rotation(pose[3:]).T@(np.asarray(state['bottle_functional_pose'][:3])-pose[:3])

    def execute_stage(self, groups):
        # This task controls one arm. Split at native action boundaries without
        # adding simulation steps, preserving the original per-action controller.
        if len(groups)!=1:raise ValueError('Diagnostic runner requires a single-arm stage')
        for action in groups[0]['actions']:
            self.action_index += 1
            before = self.sample(store=False)
            self.original_move(*deserialize([{'arm':groups[0]['arm'],'actions':[action]}]))
            after = self.sample()
            self.actions.append({'stage':self.stage, 'action_index':self.action_index,
                                 'request':copy.deepcopy(action), 'before':before, 'after':after})
            if not self.native.plan_success:break

    def execute_program(self, program):
        from envs.utils.action import ArmTag
        self.actions=[]; self.action_index=-1
        n=self.native; arm=ArmTag(self.arm)
        if self.mode not in ('frozen','adaptive','object_position'):raise ValueError('Unknown continuation')
        for self.stage, groups in enumerate(program):
            if self.mode!='frozen' and self.stage==0:
                checkpoint=self.grasp_checkpoint
                if checkpoint is None or checkpoint['delta']!=self.current_delta:
                    raise ValueError('Adaptive branch requires the matched frozen grasp checkpoint')
                n.scene.physx_system.unpack(checkpoint['physics'])
                for joint,position,velocity in checkpoint['drives']:
                    joint.set_drive_target(position);joint.set_drive_velocity_target(velocity)
                n.robot.left_gripper_val,n.robot.right_gripper_val=checkpoint['openings']
                self.samples=copy.deepcopy(checkpoint['samples'])
                self.actions=copy.deepcopy(checkpoint['actions']);self.action_index=self.actions[-1]['action_index']
                state=self.sample(store=False)
                for key in ('left_pose','right_pose','bottle_pose'):
                    if not np.allclose(state[key],self.samples[-1][key],atol=1e-7,rtol=0):
                        raise RuntimeError(f'Grasp branch restore mismatch at {key}')
                for name,joint in state['joints'].items():
                    expected=self.samples[-1]['joints'][name]
                    if any(abs(joint[key]-expected[key])>1e-7 for key in ('qpos','target')):
                        raise RuntimeError(f'Grasp branch joint restore mismatch at {name}')
                continue
            if self.mode=='adaptive' and self.stage==1:
                groups=serialize([n.move_by_displacement(arm_tag=arm,z=.1,move_axis='arm')])
            elif self.mode=='adaptive' and self.stage==2:
                groups=serialize([n.place_actor(n.bottle,arm_tag=arm,
                    target_pose=getattr(n,self.arm+'_target_pose'),functional_point_id=0,pre_dis=0.,is_open=False)])
            elif self.mode=='object_position' and self.stage==2:
                state=self.sample(store=False);pose=state[self.arm+'_pose']
                relative=rotation(pose[3:]).T@(np.asarray(state['bottle_functional_pose'][:3])-pose[:3])
                groups=copy.deepcopy(groups)
                for group in groups:
                    for action in group['actions']:
                        if action['action']=='move':
                            action['target_pose']=retarget_position(action['target_pose'],self.reference_object_in_eef,relative)
            self.execute_stage(groups)
            if self.stage==0 and n.plan_success:
                self.grasp_checkpoint={
                    'delta':copy.deepcopy(self.current_delta),'trial':len(self.results),
                    'physics':n.scene.physx_system.pack(),
                    'drives':[(j,np.array(j.get_drive_target()).copy(),np.array(j.get_drive_velocity_target()).copy())
                              for art in n.scene.get_all_articulations() for j in art.get_active_joints()],
                    'openings':(n.robot.left_gripper_val,n.robot.right_gripper_val),
                    'samples':copy.deepcopy(self.samples),'actions':copy.deepcopy(self.actions)}
            if not n.plan_success:break

    def trial(self, delta, phase):
        self.current_delta=delta
        return super().trial(delta,phase)

    def trial_details(self):
        return {'continuation':self.mode,'action_diagnostics':self.actions,
                'grasp_branch_source_trial':self.grasp_checkpoint['trial'] if self.grasp_checkpoint else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program-from',type=Path,required=True)
    p.add_argument('--cases-json',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--continuations',nargs='+',choices=('adaptive','object_position'),default=['adaptive'])
    args=p.parse_args(); output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    source=args.program_from.resolve(); cases=json.loads(args.cases_json.resolve().read_text())
    if args.repeats<1:raise ValueError('At least one repetition is required')
    if any('terminal_z' in delta for delta in cases):
        raise ValueError('Matched grasp experiment cannot calibrate a shared frozen/adaptive endpoint offset')
    manifest=EpisodeManifest(args.manifest.resolve(),'adjust_bottle'); entry=manifest.entry(0,args.index)
    if source_commit()!=manifest.source_commit:raise ValueError('Source differs from manifest')
    meta={'status':'initializing','seed':entry['seed'],'collection':'clean','source':str(source),
          'driver_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'cases':cases,'repeats':args.repeats,'continuations':args.continuations,'started_unix':time.time(),
          'branching':'Same post-grasp physics snapshot; native lift/place recomputed only in adaptive branch'}
    write(output/'run.json',meta); native=None
    try:
        initialize_runtime()
        native=importlib.import_module('envs.adjust_bottle').adjust_bottle()
        native.setup_demo(now_ep_num=args.index,seed=entry['seed'],is_test=False,**native_config('adjust_bottle',0))
        native.set_instruction(entry['instruction'])
        exp=DiagnosticExperiment(native,output,entry['seed']);exp.load_program(source)
        exp.baselines()
        meta.update(status='running',arm=exp.arm,model_id=int(native.model_id));write(output/'run.json',meta)
        for repeat in range(args.repeats):
            for delta in cases:
                for mode in ['frozen',*args.continuations]:
                    exp.mode=mode;exp.trial(delta,f'paired_{repeat}')
        exp.mode='frozen'
        if exp.trial({},'baseline_after')['outcome']!='success':raise RuntimeError('Final baseline failed')
        meta.update(status='completed',trials=len(exp.results))
    except BaseException as exc:
        meta.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if native is not None:native.close_env()
        meta['finished_unix']=time.time();write(output/'run.json',meta)


if __name__=='__main__':main()
