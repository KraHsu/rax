"""AST-only source contract. Never import audited solver modules."""
import ast
from pathlib import Path
from contracts import OPTIONS,digest

MOTION_DEFAULTS={'num_ik_seeds':32,'position_threshold':.005,'rotation_threshold':.05,'use_es_ik':None,
'es_ik_learning_rate':1.0,'ik_seed':1531,'store_ik_debug':False,'ik_particle_opt':True,
'ik_opt_iters':None,'use_ik_fixed_samples':None,'project_pose_to_goal_frame':True,'high_precision':False,
'collision_activation_distance':None,'sync_cuda_time':None}
MAP={'num_ik_seeds':'num_seeds','es_ik_learning_rate':'es_learning_rate','ik_seed':'seed',
'use_es_ik':'use_es','store_ik_debug':'store_debug','ik_particle_opt':'use_particle_opt',
'ik_opt_iters':'grad_iters','use_ik_fixed_samples':'use_fixed_samples'}

def check(package):
    root=Path(package)/'design/source';m=ast.parse((root/'7_motion_gen.py').read_text(encoding='utf-8'))
    cls=next(x for x in m.body if isinstance(x,ast.ClassDef) and x.name=='MotionGenConfig')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='load_from_robot_config')
    defaults={a.arg:d for a,d in zip(fn.args.args[-len(fn.args.defaults):],fn.args.defaults)}
    for key,val in MOTION_DEFAULTS.items():
        if ast.literal_eval(defaults[key])!=val or OPTIONS[MAP.get(key,key)]!=val:raise ValueError('Effective MotionGen IK default changed: '+key)
    call=next(x for x in ast.walk(fn) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and isinstance(x.func.value,ast.Name) and x.func.value.id=='IKSolverConfig' and x.func.attr=='load_from_robot_config')
    if [ast.unparse(x) for x in call.args]!=['robot_cfg','world_model','tensor_args','num_ik_seeds','position_threshold','rotation_threshold','world_coll_checker','base_config_data','particle_ik_file','gradient_ik_file']:raise ValueError('MotionGen IK positional mapping changed')
    expected={'use_cuda_graph':'use_cuda_graph','self_collision_check':'self_collision_check','self_collision_opt':'self_collision_opt','grad_iters':'ik_opt_iters','use_particle_opt':'ik_particle_opt','sync_cuda_time':'sync_cuda_time','use_es':'use_es_ik','es_learning_rate':'es_ik_learning_rate','use_fixed_samples':'use_ik_fixed_samples','store_debug':'store_ik_debug','collision_activation_distance':'collision_activation_distance','project_pose_to_goal_frame':'project_pose_to_goal_frame','seed':'ik_seed','high_precision':'high_precision'}
    if {x.arg:ast.unparse(x.value) for x in call.keywords}!=expected:raise ValueError('MotionGen IK keyword mapping changed')
    # Source success uses <=; rotation is source metric, not a geodesic angle.
    tree=ast.parse((root/'6_ik_solver.py').read_text(encoding='utf-8'))
    success=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='get_success')
    comparisons=[ast.unparse(x) for x in ast.walk(success) if isinstance(x,ast.Compare)]
    if comparisons!=['position_error <= position_threshold','rotation_error <= rotation_threshold']:raise ValueError('IK convergence semantics changed')
    return dict(mode='AST_only_no_curobo_import',motion_source_sha256=digest(root/'7_motion_gen.py'),
                ik_source_sha256=digest(root/'6_ik_solver.py'),native_effective_defaults=MOTION_DEFAULTS,
                explicit_runtime_differences=['CUDA graph disabled','return32 instead of original selected count','fixed collision modes','no graph/trajopt/attempt schedule'],
                actual_cuda_constructor_tested=False)
