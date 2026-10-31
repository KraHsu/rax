import ast
import copy
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
P=Path(__file__).parent/'package'
sys.path.insert(0,str(P/'runtime'))
import driver,contracts,geometry,gpu_backend


def raw(target,success=True):
    q=np.tile(np.array(target['start_joints_rad'],dtype=np.float32).astype(float),(32,1))
    T=geometry.fk(P/'design/source/1_arx5_description_isaac.urdf',q)
    return dict(solution=q[None].tolist(),generated_seeds=q[:,None].tolist(),success=[[bool(success and np.linalg.norm(T[0,:3,3]-target['position_m'])<=.005)]*32],
                position_error=[np.linalg.norm(T[:,:3,3]-target['position_m'],axis=1).tolist()],rotation_error=[[0.]*32],total_error=[[0.]*32],solve_time=.01,solve_wall_seconds_including_result_transfer=.02,result_dtype='torch.float32',target_cast_dtype='torch.float32',
                constraint_feasible=[[True]]*32,constraint_value=[[0.]]*32,constraint_components='aggregate only',fk_position=T[:,:3,3].tolist(),
                fk_quaternion=[geometry.matrix_quaternion(r).tolist() for r in T[:,:3,:3]])


@pytest.fixture
def setup(tmp_path,monkeypatch):
    original=contracts.inputs
    monkeypatch.setattr(driver,'inputs',lambda package,external:original(package,False))
    class Fake:
        calls=0;close_calls=0;fail_at=None;false_at=None;double=False
        def __init__(self,*args):pass
        def prepare(self,mode):
            contracts.atomic(tmp_path/'out/modes'/f'{mode["id"]}.json',dict(mode=mode))
        def solve(self,target,charged):
            self.calls+=1
            contracts.atomic(tmp_path/'out/solver_entries.json',dict(solve_single_started=self.calls,unexpected_entries=0))
            if self.calls==self.fail_at:raise RuntimeError('original-solver-error')
            charged('constraint_evaluations')
            if self.double:charged('constraint_evaluations')
            charged('fk_evaluations')
            result=raw(target,self.calls!=self.false_at)
            folder=tmp_path/'out/slots'/f'ik_{self.calls-1:02d}'
            contracts.atomic(folder/'seed_matrix.json',result['generated_seeds'])
            contracts.atomic(folder/'solver_result.json',{k:v for k,v in result.items() if not k.startswith(('constraint','fk_'))})
            contracts.atomic(folder/'post_constraints.json',{k:v for k,v in result.items() if k.startswith('constraint')})
            return result
        def close(self):self.close_calls+=1
    obj=Fake()
    def factory(*args):
        contracts.atomic(tmp_path/'out/runtime_identity.json',dict(device='cuda:0',visible_devices='0',scenes=0,native_task_imports=False))
        return obj
    return tmp_path/'out',obj,factory


def test_complete_fixed_twelve(setup):
    out,obj,factory=setup;s=driver.execute(P,out,factory)
    assert s['status']=='completed_diagnostic' and obj.calls==12 and obj.close_calls==1
    assert s['counts']==dict(solve_attempts=12,constraint_evaluations=12,fk_evaluations=12)
    assert len(s['records'])==12 and s['native_initializations']==0
    assert driver.verify_output(out)['status']=='completed_diagnostic'


def test_fk_control_blocks_without_replacement(setup):
    out,obj,factory=setup;obj.false_at=1;s=driver.execute(P,out,factory)
    assert obj.calls==1 and s['status']=='blocked_known_fk_control'
    assert driver.verify_output(out)['status']=='blocked_known_fk_control'


def test_later_ik_failure_is_observation(setup):
    out,obj,factory=setup;obj.false_at=2;s=driver.execute(P,out,factory)
    assert obj.calls==12 and s['status']=='completed_diagnostic'
    assert not contracts.read(out/'slots/ik_01/record.json')['report']['any_reported_success']


def test_exception_retains_prefix_no_retry(setup):
    out,obj,factory=setup;obj.fail_at=3
    with pytest.raises(RuntimeError,match='original'):driver.execute(P,out,factory)
    s=contracts.read(out/'state.json');assert s['status']=='incomplete' and s['counts']['solve_attempts']==3 and len(s['records'])==2
    assert (out/'slots/ik_02/intent.json').exists() and obj.close_calls==1
    with pytest.raises(FileExistsError):driver.execute(P,out,factory)
    assert obj.calls==3


def test_post_budget_rejects_second_eval(setup):
    out,obj,factory=setup;obj.double=True
    with pytest.raises(RuntimeError,match='budget'):driver.execute(P,out,factory)
    assert contracts.read(out/'state.json')['counts']==dict(solve_attempts=1,constraint_evaluations=1,fk_evaluations=0)


def test_deadline_no_solve(setup):
    out,obj,factory=setup;times=iter([0,901])
    with pytest.raises(TimeoutError):driver.execute(P,out,factory,clock=lambda:next(times))
    assert obj.calls==0 and obj.close_calls==1

@pytest.mark.parametrize('field,value',[('solution',[]),('generated_seeds',[]),('success',[[1]*32]),('position_error',[[float('nan')]*32]),('fk_position',[[0,0,0]]*32)])
def test_invalid_raw_rejected(field,value):
    _,targets,_=contracts.inputs(P,False);t=next(iter(targets.values()));r=raw(t);r[field]=value
    with pytest.raises(ValueError):contracts.analyze(P,t,r)


def test_source_threshold_float32_and_equality():
    _,targets,_=contracts.inputs(P,False);t=next(iter(targets.values()));r=raw(t)
    r['rotation_error']=[[float(np.float32(.05))]*32]
    assert contracts.analyze(P,t,r)['reported_success_count']==32
    r['rotation_error'][0][0]=float(np.nextafter(np.float32(.05),np.float32(1)))
    with pytest.raises(ValueError,match='threshold'):contracts.analyze(P,t,r)


def test_seed_zero_bound_to_recorded_float32():
    _,targets,_=contracts.inputs(P,False);t=next(iter(targets.values()));r=raw(t);r['generated_seeds'][0][0][0]+=.001
    with pytest.raises(ValueError,match='Seed0'):contracts.analyze(P,t,r)


def test_cpu_imports_never_import_solver():
    code=f"import sys;sys.path.insert(0,{str(P/'runtime')!r});import driver,gpu_backend;driver.preflight({str(P)!r},False);assert not any(x.split('.')[0] in ('torch','curobo','sapien','envs') for x in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_original_ast_seed_preserves_tensor_and_inputs():
    # Execute only frozen get_seed body with CPU Torch; no cuRobo module import.
    import torch
    tree=ast.parse((P/'design/source/6_ik_solver.py').read_text());cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='IKSolver')
    fn=copy.deepcopy(next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='get_seed'))
    fn.decorator_list=[];fn.returns=None
    for a in fn.args.args:a.annotation=None
    ns={'torch':torch};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'frozen_ast','exec'),ns)
    class Stub:
        dof=6
        calls=0
        def generate_seed(self,num_seeds,**kw):
            self.calls+=1;assert num_seeds==31;return torch.ones((1,num_seeds,6))
    from types import SimpleNamespace
    obj=Stub();q=torch.zeros((1,1,6));before=q.clone();out=ns['get_seed'](obj,32,SimpleNamespace(batch=1),True,q)
    assert obj.calls==1 and tuple(out.shape)==(32,1,6) and torch.equal(q,before) and torch.equal(out[0],q[0])


def test_guard_blocks_constructor_or_extra_entry(tmp_path):
    class IK:
        calls=0
        def solve_single(self,*a,**kw):self.calls+=1;return object()
        def solve_batch(self,*a,**kw):pytest.fail('Forbidden underlying call')
        def get_seed(self,*a,**kw):return 'seed-object'
    b=gpu_backend.Backend.__new__(gpu_backend.Backend);b.IK=IK;b.originals={};b.output=tmp_path;b.allowed=False;b.calls=0;b.seeds=None;b.solver=IK()
    b._install_guards()
    try:
        with pytest.raises(RuntimeError):b.solver.solve_single()
        assert b.solver.calls==0
        b.allowed=True;result=b.solver.solve_single();assert result is not None and b.calls==1 and b.solver.calls==1
        with pytest.raises(RuntimeError):b.solver.solve_single()
        b.allowed=True
        with pytest.raises(RuntimeError):b.solver.solve_batch()
        b.calls=12
        with pytest.raises(RuntimeError):b.solver.solve_single()
    finally:b.close()
    assert 'solve_single' not in b.originals


def test_raw_tamper_rejected(setup):
    out,obj,factory=setup;driver.execute(P,out,factory)
    f=out/'slots/ik_00/raw.json';r=contracts.read(f);r['solution'][0][0][0]+=.1;contracts.atomic(f,r)
    with pytest.raises(ValueError,match='SHA'):driver.verify_output(out)


def test_extra_record_rejected(setup):
    out,obj,factory=setup;driver.execute(P,out,factory);contracts.atomic(out/'slots/ik_12/record.json',{})
    with pytest.raises(ValueError,match='record set'):driver.verify_output(out)


def test_initialization_exception_recorded_without_solve(setup):
    out,obj,_=setup
    def factory(*a):raise RuntimeError('import/constructor original')
    with pytest.raises(RuntimeError,match='constructor'):driver.execute(P,out,factory)
    state=contracts.read(out/'state.json')
    assert state['counts']['solve_attempts']==0 and state['status']=='incomplete'


def test_cleanup_failure_keeps_original_exception(setup):
    out,obj,factory=setup;obj.fail_at=1
    def close():raise ValueError('cleanup failed')
    obj.close=close
    with pytest.raises(RuntimeError,match='original-solver-error'):driver.execute(P,out,factory)
    state=contracts.read(out/'state.json')
    assert state['error']['type']=='RuntimeError' and state['cleanup_error']['type']=='ValueError'


def test_nonfinite_tensor_retained_bounded(tmp_path):
    import torch
    b=gpu_backend.Backend.__new__(gpu_backend.Backend);b.torch=torch;b.output=tmp_path
    with pytest.raises(ValueError):b._tensor(torch.tensor([float('nan')]),(1,),'failure')
    assert contracts.read(tmp_path/'invalid_tensor.json')['values']==['nan']


def test_incorrect_success_position_claim_rejected():
    _,targets,_=contracts.inputs(P,False);t=targets['original_preplace'];r=raw(t)
    r['position_error']=[[0.]*32];r['success']=[[True]*32]
    with pytest.raises(ValueError,match='independent FK'):contracts.analyze(P,t,r)

@pytest.mark.parametrize('name',['seed_matrix.json','solver_result.json','post_constraints.json','intent.json'])
def test_missing_original_component_is_not_complete(setup,name):
    out,obj,factory=setup;driver.execute(P,out,factory);(out/'slots/ik_00'/name).unlink()
    with pytest.raises((ValueError,FileNotFoundError)):driver.verify_output(out)


def test_rehashed_wrong_target_intent_rejected(setup):
    out,obj,factory=setup;driver.execute(P,out,factory);folder=out/'slots/ik_00'
    intent=contracts.read(folder/'intent.json');intent['target']['position_m'][0]+=.001;contracts.atomic(folder/'intent.json',intent)
    record=contracts.read(folder/'record.json');record['evidence_sha256']['intent.json']=contracts.digest(folder/'intent.json');contracts.atomic(folder/'record.json',record)
    state=contracts.read(out/'state.json');state['records'][0]['record_sha256']=contracts.digest(folder/'record.json');contracts.atomic(out/'state.json',state)
    with pytest.raises(ValueError,match='intent'):driver.verify_output(out)


def test_source_hash_change_cpu_fail_closed(tmp_path):
    import shutil
    q=tmp_path/'package';shutil.copytree(P,q)
    with (q/'design/source/6_ik_solver.py').open('a') as f:f.write('\n# changed\n')
    with pytest.raises(ValueError,match='source drift'):contracts.inputs(q,False)


def test_no_native_label_can_be_promoted(setup):
    out,obj,factory=setup;driver.execute(P,out,factory);state=contracts.read(out/'state.json');state['native_task_label_available']=True;contracts.atomic(out/'state.json',state)
    with pytest.raises(ValueError,match='acceptance label'):driver.verify_output(out)


def test_exact_tensor_type_and_dtype(tmp_path):
    import torch
    b=gpu_backend.Backend.__new__(gpu_backend.Backend);b.torch=torch;b.output=tmp_path
    with pytest.raises(ValueError,match='dtype'):b._tensor(torch.zeros(1,dtype=torch.float64),(1,),'wrong')
    with pytest.raises(ValueError,match='exact tensor'):b._tensor(np.zeros(1),(1,),'wrong')


def test_aggregate_constraint_result_must_match_boolean():
    _,targets,_=contracts.inputs(P,False);t=next(iter(targets.values()));r=raw(t);r['constraint_value'][0][0]=1
    with pytest.raises(ValueError,match='feasibility'):contracts.analyze(P,t,r)


def test_prepare_real_wrapper_bounds_hierarchy(tmp_path):
    # Exact source uses IKSolver.solver (WrapBase).safety_rollout, not a direct IK field.
    import torch,yaml
    from types import SimpleNamespace as S
    tree=ast.parse((P/'design/source/6_ik_solver.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='IKSolver')
    init=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
    attrs={ast.unparse(x) for x in ast.walk(init) if isinstance(x,ast.Attribute)}
    assert 'self.solver.safety_rollout.action_bound_lows' in attrs
    _,_,limits=contracts.inputs(P,False)
    effective=S(base_link='fr_base_link',ee_link='fr_link6',lock_joints={'fr_joint7':.04,'fr_joint8':.04})
    bounds=torch.tensor(limits,dtype=torch.float32)
    kin=S(generator_config=effective,get_joint_limits=lambda:S(position=bounds))
    solver=S(dof=6,joint_names=contracts.JOINTS,num_seeds=32,seed=1531,position_threshold=.005,rotation_threshold=.05,ik_nn_seeder=None,
             kinematics=kin,solver=S(safety_rollout=S(action_bound_lows=bounds[0],action_bound_highs=bounds[1])))
    assert not hasattr(solver,'safety_rollout')
    b=gpu_backend.Backend.__new__(gpu_backend.Backend);b.package=P;b.output=tmp_path;b.limits=limits;b.solver=None;b.mode=None;b.calls=0
    b.torch=S(cuda=S(empty_cache=lambda:None));b.tensor_args=S();b.load_yaml=lambda p:yaml.safe_load(Path(p).read_text())
    configs=[]
    def config(cfg,**kw):configs.append((cfg,kw));return cfg
    b.Config=S(load_from_robot_config=config);b.IK=lambda config:solver
    b._tensor=lambda value,shape,name:gpu_backend.Backend._tensor(S(torch=torch,output=tmp_path),value,shape,name)
    mode=contracts.inputs(P,False)[0]['modes'][0]
    b.prepare(mode)
    assert b.solver is solver and len(configs)==1
    assert configs[0][1]['es_learning_rate']==1.0 and configs[0][1]['num_seeds']==32
    assert contracts.read(tmp_path/'modes/source_table_and_self.json')['sampling_bounds']==limits.tolist()
    b.prepare(mode);assert len(configs)==1

@pytest.mark.parametrize('field',['error','cleanup_error'])
def test_relabelled_failed_or_cleanup_error_state_rejected(setup,field):
    out,obj,factory=setup;driver.execute(P,out,factory)
    state=contracts.read(out/'state.json');state[field]={'type':'RuntimeError','message':'previously failed'};contracts.atomic(out/'state.json',state)
    with pytest.raises(ValueError,match='cleanup-error'):driver.verify_output(out)


def test_blocked_control_cannot_hide_later_uncommitted_attempt(setup):
    out,obj,factory=setup;obj.false_at=1;driver.execute(P,out,factory)
    contracts.atomic(out/'slots/ik_01/intent.json',{'attempt':'hidden'})
    with pytest.raises(ValueError,match='directory set'):driver.verify_output(out)
