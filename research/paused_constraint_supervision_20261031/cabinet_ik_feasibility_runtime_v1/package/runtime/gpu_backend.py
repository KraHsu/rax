"""Lazy GPU-only backend. Import itself is CPU safe; Backend construction requires owned execution."""
import copy
import inspect
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from contracts import OPTIONS,JOINTS,read,digest,atomic,array


class Backend:
    def __init__(self,package,output,limits):
        self.package=Path(package);self.output=Path(output);self.limits=limits
        self.originals={};self.solver=None;self.mode=None;self.calls=0;self.allowed=False;self.seeds=None
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='0':raise ValueError('Reviewed GPU0 visibility required')
        plan=read(self.package/'plan.json')
        for p,h in plan['external_files'].items():
            if digest(p)!=h:raise ValueError('Source/dependency drift before GPU import: '+p)
        # GPU import begins here, only in a child with the shared lease inherited from owner.
        import torch
        if np.__version__!='1.26.4' or torch.__version__!='2.4.1+cu121' or torch.version.cuda!='12.1':raise ValueError('Frozen native numerical runtime changed')
        from curobo.wrap.reacher.ik_solver import IKSolver,IKSolverConfig,IKResult
        from curobo.types.base import TensorDeviceType
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.util_file import load_yaml
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelState
        from curobo.rollout.rollout_base import RolloutMetrics
        self.torch=torch;self.IK=IKSolver;self.Config=IKSolverConfig;self.Result=IKResult
        self.Pose=Pose;self.JointState=JointState;self.FKState=CudaRobotModelState;self.Metrics=RolloutMetrics;self.load_yaml=load_yaml
        for cls in (IKSolver,IKSolverConfig,IKResult,Pose,JointState,CudaRobotModelState,RolloutMetrics):
            p=str(Path(inspect.getfile(cls)).resolve())
            if p not in plan['external_files'] or digest(p)!=plan['external_files'][p]:raise ValueError('Actual class source is outside frozen closure: '+p)
        # Record loaded CUDA extension identities, all must already be frozen. No JIT replacement accepted.
        loaded={}
        for name,module in list(sys.modules.items()):
            p=getattr(module,'__file__',None)
            if name.startswith('curobo.') and p:
                p=str(Path(p).resolve())
                if p not in plan['external_files'] or digest(p)!=plan['external_files'][p]:raise ValueError('Unfrozen loaded cuRobo module: '+p)
                loaded[name]=dict(path=p,sha256=plan['external_files'][p])
        self.tensor_args=TensorDeviceType(device=torch.device('cuda:0'),dtype=torch.float32)
        atomic(self.output/'runtime_identity.json',dict(pid=os.getpid(),ppid=os.getppid(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
             device=str(self.tensor_args.device),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
             device_name=torch.cuda.get_device_name(0),loaded_curobo_modules=loaded,
             explicit_device_to_cpu_transfers=True,native_task_imports=False,scenes=0))
        self._install_guards()

    def _install_guards(self):
        # Guard every public solve entry before any IKSolver construction. Internal optimization
        # methods remain original. An unexpected entry is blocked before executing the original.
        for name in list(vars(self.IK)):
            if not name.startswith('solve'):continue
            original=getattr(self.IK,name);self.originals[name]=original
            def guard(instance,*args,_name=name,_original=original,**kwargs):
                if _name!='solve_single' or not self.allowed or instance is not self.solver or self.calls>=12:
                    atomic(self.output/'unexpected_solver_entry.json',dict(method=_name,completed_entries=self.calls,blocked_before_call=True))
                    raise RuntimeError('Unbudgeted solver entry')
                self.allowed=False;self.calls+=1
                atomic(self.output/'solver_entries.json',dict(solve_single_started=self.calls,unexpected_entries=0))
                return _original(instance,*args,**kwargs)
            setattr(self.IK,name,guard)
        original=self.IK.get_seed;self.originals['get_seed']=original
        def seed_guard(instance,*args,**kwargs):
            if instance is not self.solver or self.seeds is not None:raise RuntimeError('Seed generation outside unique solve')
            out=original(instance,*args,**kwargs)  # exactly one original call; no RNG replay
            self.seeds=self._tensor(out,(32,1,6),'generated_seeds')
            atomic(self.output/'slots'/f'ik_{self.calls-1:02d}'/'seed_matrix.json',self.seeds)
            return out
        self.IK.get_seed=seed_guard

    def _tensor(self,value,shape,name,boolean=False):
        t=self.torch
        if type(value)is not t.Tensor or tuple(value.shape)!=tuple(shape):raise ValueError('Unexpected exact tensor/shape: '+name)
        if value.dtype!=(t.bool if boolean else t.float32):raise ValueError('Unexpected tensor dtype: '+name)
        result=value.detach().cpu().tolist()
        try:array(result,shape,name,boolean)
        except ValueError:
            def finite_json(x):
                if isinstance(x,list):return [finite_json(v) for v in x]
                return x if isinstance(x,bool) or math.isfinite(x) else str(x)
            atomic(self.output/'invalid_tensor.json',dict(field=name,shape=list(shape),dtype=str(value.dtype),finite=False,values=finite_json(result)))
            raise
        return result

    def prepare(self,mode):
        if self.mode==mode['id']:return
        self.solver=None;self.mode=mode['id'];self.torch.cuda.empty_cache();mode_started=time.monotonic()
        cfg=copy.deepcopy(self.load_yaml(str(self.package/'design/source/0_curobo_right.yml'))['robot_cfg'])
        k=cfg['kinematics']
        if (k['base_link']!='fr_base_link' or k['ee_link']!='fr_link6' or k['lock_joints']!={'fr_joint7':.04,'fr_joint8':.04}):raise ValueError('Unexpected native robot frame/locks')
        options=dict(OPTIONS,self_collision_check=mode['self_collision_check'],self_collision_opt=mode['self_collision_opt'])
        options.update(base_cfg_file=str(self.package/'design/source/base_cfg.yml'),particle_file=str(self.package/'design/source/particle_ik.yml'),gradient_file=str(self.package/'design/source/gradient_ik_autotune.yml'))
        config=self.Config.load_from_robot_config(cfg,world_model=copy.deepcopy(mode['world_model']),tensor_args=self.tensor_args,**options)
        solver=self.IK(config)
        if (solver.dof!=6 or solver.joint_names!=JOINTS or solver.num_seeds!=32 or solver.seed!=1531
            or solver.position_threshold!=.005 or solver.rotation_threshold!=.05 or solver.ik_nn_seeder is not None):raise ValueError('Runtime IK configuration differs')
        gc=solver.kinematics.generator_config
        if gc.base_link!='fr_base_link' or gc.ee_link!='fr_link6' or gc.lock_joints!={'fr_joint7':.04,'fr_joint8':.04}:raise ValueError('Effective robot frame/lock mismatch')
        lim=self._tensor(solver.kinematics.get_joint_limits().position,(2,6),'joint_limits')
        if not np.array_equal(np.array(lim),self.limits.astype(np.float32).astype(float)):raise ValueError('Effective bounds differ from frozen XML')
        lower=self._tensor(solver.solver.safety_rollout.action_bound_lows,(6,),'sampling_lower_bounds')
        upper=self._tensor(solver.solver.safety_rollout.action_bound_highs,(6,),'sampling_upper_bounds')
        if [lower,upper]!=lim:raise ValueError('Sampling/action bounds differ from effective joints')
        self.solver=solver
        atomic(self.output/'modes'/f'{self.mode}.json',dict(mode=mode,options=options,joint_names=solver.joint_names,
              limits=lim,sampling_bounds=[lower,upper],lock_joints=gc.lock_joints,base_link=gc.base_link,ee_link=gc.ee_link,
              calls_before_mode=self.calls,configuration_wall_seconds=time.monotonic()-mode_started,world_model_includes_cabinet=False))

    def solve(self,target,charged):
        t=self.torch;self.seeds=None
        goal=self.Pose(position=t.tensor([target['position_m']],**self.tensor_args.as_torch_dict()),
                       quaternion=t.tensor([target['runtime_quaternion_wxyz']],**self.tensor_args.as_torch_dict()))
        q=t.tensor([target['start_joints_rad']],**self.tensor_args.as_torch_dict())
        self.solver.reset_seed();self.allowed=True
        start=time.monotonic()
        result=self.solver.solve_single(goal,retract_config=q,seed_config=q.unsqueeze(1),return_seeds=32,num_seeds=32,use_nn_seed=True,newton_iters=None,link_poses=None)
        self.allowed=False
        if type(result)is not self.Result or self.seeds is None:raise ValueError('Unexpected result/seed observation')
        if type(result.solve_time) not in (int,float):raise ValueError('Unexpected solve_time type')
        out=dict(solution=self._tensor(result.solution,(1,32,6),'solution'),
                 success=self._tensor(result.success,(1,32),'success',True),
                 position_error=self._tensor(result.position_error,(1,32),'position_error'),
                 rotation_error=self._tensor(result.rotation_error,(1,32),'rotation_error'),
                 total_error=self._tensor(result.error,(1,32),'total_error'),
                 generated_seeds=self.seeds,solve_time=float(result.solve_time),
                 solve_wall_seconds_including_result_transfer=time.monotonic()-start,
                 result_dtype='torch.float32',target_cast_dtype='torch.float32')
        folder=self.output/'slots'/f'ik_{self.calls-1:02d}'
        atomic(folder/'solver_result.json',out)  # survives later evaluation exceptions
        solutions=result.solution.reshape(32,6)
        charged('constraint_evaluations')
        metrics=self.solver.check_constraints(self.JointState(position=solutions,joint_names=JOINTS))
        if type(metrics)is not self.Metrics:raise ValueError('Unexpected constraint metrics class')
        out['constraint_feasible']=self._tensor(metrics.feasible,(32,1),'constraint feasible',True)
        out['constraint_value']=None if metrics.constraint is None else self._tensor(metrics.constraint,(32,1),'constraint value')
        out['constraint_components']='aggregate_only; per collision source decomposition unavailable in returned RolloutMetrics'
        atomic(folder/'post_constraints.json',{k:v for k,v in out.items() if k.startswith('constraint')})
        charged('fk_evaluations'); state=self.solver.fk(solutions)
        if type(state)is not self.FKState:raise ValueError('Unexpected FK state class')
        out['fk_position']=self._tensor(state.ee_position,(32,3),'FK position')
        out['fk_quaternion']=self._tensor(state.ee_quaternion,(32,4),'FK quaternion')
        return out

    def close(self):
        self.allowed=False
        for name,value in self.originals.items():setattr(self.IK,name,value)
        self.originals.clear();self.solver=None
