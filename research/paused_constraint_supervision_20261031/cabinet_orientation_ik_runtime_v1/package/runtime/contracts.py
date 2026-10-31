"""CPU-only fixed IK experiment contract and independently recomputable evidence."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from geometry import fk, quaternion, matrix_quaternion, angular_error

VERSION='cabinet_orientation_ik_runtime_v1'
DESIGN_SHA='664e36c018069453228fa88ee898f54ed4b32fe9f91f015582a4b53b28a5abbf'
TARGETS_SHA='5bfe52fdf855c87b80e003ce4b91161018261335fe4f93a1d5894392c7d8d90c'
JOINTS=['fr_joint'+str(i) for i in range(1,7)]
FLAGS={k:False for k in ('native_task_label_available','formal_evidence_eligible','acceptance_ready','publication_allowed','training_allowed')}
SOLVER_FIELDS={'solution','success','position_error','rotation_error','total_error','generated_seeds','solve_time','solve_wall_seconds_including_result_transfer','result_dtype','target_cast_dtype'}
CONSTRAINT_FIELDS={'constraint_feasible','constraint_value','constraint_components'}
OPTIONS=dict(num_seeds=32,position_threshold=.005,rotation_threshold=.05,use_cuda_graph=False,
 self_collision_check=True,self_collision_opt=True,grad_iters=None,use_particle_opt=True,
 sync_cuda_time=None,use_gradient_descent=False,use_es=None,es_learning_rate=1.0,
 use_fixed_samples=None,store_debug=False,regularization=True,collision_activation_distance=None,
 high_precision=False,project_pose_to_goal_frame=True,seed=1531)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def value_hash(value):return hashlib.sha256(canonical(value)).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_bytes(canonical(value)+b'\n');tmp.replace(path)
def array(value,shape,name,boolean=False):
    a=np.asarray(value)
    if a.shape!=tuple(shape) or (a.dtype.kind!='b' if boolean else a.dtype.kind not in 'fiu') or not np.isfinite(a).all():raise ValueError('Invalid finite shape/type: '+name)
    return a.astype(bool if boolean else float)
def safe_relative(root,name):
    path=(Path(root)/name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(Path(root).resolve()):raise ValueError('Package path escape')
    return path


def inputs(package,external=True):
    package=Path(package); d=package/'design'
    if digest(d/'probe_design.json')!=DESIGN_SHA or digest(d/'targets.json')!=TARGETS_SHA:raise ValueError('Frozen design/targets changed')
    design=read(d/'probe_design.json'); targets=read(d/'targets.json')
    for filename,key in [('captured_request.json','captured_request_sha256'),('geometry_report.json','geometry_report_sha256')]:
        if digest(d/filename)!=design['inputs'][key]:raise ValueError('Historical input changed')
    if len(targets)!=12 or len(design['slots'])!=12:raise ValueError('Fixed budget changed')
    from orientation_contract import verify_targets
    verify_targets(package,targets)
    old=read(d/'orientation_inputs/previous_probe_design.json')
    mode=next(x for x in old['modes'] if x['id']=='source_table_and_self')
    if design['modes']!=[mode]:raise ValueError('Original self/table collision mode must stay enabled')
    slots=[dict(slot_id=f'ik_{i:02d}',mode='source_table_and_self',target=t['id'],solver_calls_max=1,reseeds_after_outcome=0) for i,t in enumerate(targets)]
    if design['slots']!=slots:raise ValueError('Fixed twelve-slot order/budget changed')
    source_entries=[]
    for name in design['source_manifests']:
        rows=read(d/name)
        for row in ([rows] if isinstance(rows,dict) else rows):
            if digest(safe_relative(d,row['local_file']))!=row['sha256']:raise ValueError('Copied source drift')
            if external and digest(row['path'])!=row['sha256']:raise ValueError('Installed source drift: '+row['path'])
            source_entries.append(row)
    for target in targets:
        array(target['position_m'],(3,),'target position'); R=array(target['rotation_matrix'],(3,3),'target rotation')
        if np.max(np.abs(R.T@R-np.eye(3)))>1e-6 or abs(np.linalg.det(R)-1)>1e-6:raise ValueError('Target rotation invalid')
        q=array(target['start_joints_rad'],(6,),'start joints')
        quat=target.get('recorded_quaternion_wxyz',matrix_quaternion(R).tolist())
        if angular_error(quaternion(array(quat,(4,),'target quaternion')),R)>1e-6:raise ValueError('Quaternion/frame mismatch')
        target['runtime_quaternion_wxyz']=quat
    urdf=d/'source/1_arx5_description_isaac.urdf'
    j={x.get('name'):x for x in ET.parse(urdf).getroot().findall('joint')}
    limits=np.array([[float(j[n].find('limit').get(k)) for n in JOINTS] for k in ('lower','upper')])
    for t in targets:
        q=np.array(t['start_joints_rad'])
        if not ((q>=limits[0])&(q<=limits[1])).all():raise ValueError('Start outside source limits')
    known=fk(urdf,[targets[0]['start_joints_rad']])[0]
    if np.max(np.abs(known[:3,3]-targets[0]['position_m']))>1e-12 or angular_error(known[:3,:3],np.array(targets[0]['rotation_matrix']))>1e-12:raise ValueError('Known FK control/source changed')
    return design,{t['id']:t for t in targets},limits


def analyze(package,target,result):
    # All raw fields remain retained even if a later invariant fails.
    if set(result)!=SOLVER_FIELDS|CONSTRAINT_FIELDS|{'fk_position','fk_quaternion'}:raise ValueError('Unexpected/missing raw result fields')
    if result['result_dtype']!='torch.float32' or result['target_cast_dtype']!='torch.float32':raise ValueError('Result dtype contract changed')
    array(result['total_error'],(1,32),'total error')
    if not math.isfinite(result['solve_wall_seconds_including_result_transfer']) or result['solve_wall_seconds_including_result_transfer']<0:raise ValueError('Invalid wall time')
    q=array(result['solution'],(1,32,6),'solution')[0]
    seeds=array(result['generated_seeds'],(32,1,6),'generated seeds')
    if not np.array_equal(seeds[0,0],np.array(target['start_joints_rad'],dtype=np.float32).astype(float)):raise ValueError('Seed0 differs from exact float32 supplied start')
    success=array(result['success'],(1,32),'success',True)[0]
    pos=array(result['position_error'],(1,32),'source position error')[0]
    rot=array(result['rotation_error'],(1,32),'source rotation metric')[0]
    feasible=array(result['constraint_feasible'],(32,1),'post constraint feasible',True)[:,0]
    constraint=array(result['constraint_value'],(32,1),'aggregate constraint')[:,0]
    if not np.array_equal(feasible,constraint==0):raise ValueError('Source constraint feasibility inconsistent')
    p=array(result['fk_position'],(32,3),'GPU FK position')
    quat=array(result['fk_quaternion'],(32,4),'GPU FK quaternion')
    if np.any(np.linalg.norm(quat,axis=1)<.99) or np.any(np.linalg.norm(quat,axis=1)>1.01):raise ValueError('Invalid GPU FK quaternion norm')
    T=fk(Path(package)/'design/source/1_arx5_description_isaac.urdf',q)
    dpos=np.linalg.norm(T[:,:3,3]-p,axis=1)
    dang=np.array([angular_error(a,quaternion(b)) for a,b in zip(T[:,:3,:3],quat)])
    # Deliberate implementation check tolerance, separate from source solver success thresholds.
    if max(dpos)>1e-5 or max(dang)>1e-4:raise ValueError('GPU FK disagrees with independent XML FK')
    errp=np.linalg.norm(T[:,:3,3]-target['position_m'],axis=1)
    erra=np.array([angular_error(x,np.array(target['rotation_matrix'])) for x in T[:,:3,:3]])
    if np.any(pos<0) or np.any(rot<0) or not math.isfinite(result['solve_time']) or result['solve_time']<0:raise ValueError('Invalid solver residual/time')
    if np.max(np.abs(pos-errp))>1e-5:raise ValueError('Source position error disagrees with independent FK')
    # Preserve reported success independently; post evaluation is a fresh constraint check, not a relabel.
    if np.any(success & ((pos>float(np.float32(.005)))|(rot>float(np.float32(.05))))):raise ValueError('Reported success violates source thresholds')
    from orientation_contract import analyze_world
    return dict(reported_success_count=int(success.sum()),post_constraint_feasible_count=int(feasible.sum()),
        any_reported_success=bool(success.any()),seed_matrix_sha256=value_hash(result['generated_seeds']),
        xml_goal_position_error_m=errp.tolist(),xml_goal_geodesic_angle_rad=erra.tolist(),
        gpu_xml_position_error_m=dpos.tolist(),gpu_xml_geodesic_angle_rad=dang.tolist(),
        source_rotation_error_units='source_pose_cost_metric_not_geodesic_radians',
        all_solutions_preserved=True,reference_world_geometry=analyze_world(target,T),**FLAGS)
