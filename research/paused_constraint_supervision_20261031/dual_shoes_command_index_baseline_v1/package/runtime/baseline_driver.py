"""Finite qualification of a captured EEF command sequence, never timed expert replay."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'code/rax'))
import numpy as np
from services.robotwin.collection import atomic_json, initial_scene, scene_fingerprint
from services.robotwin.acceptance import derived_values_equal, trajectory_violations
from services.robotwin.acceptance_replay import (arrays, sample, samples_to_trajectory, replay,
    tracking_report, install_base_frame_adapter, ALIGNMENT, SUCCESS_RULE, TRACKING, command_rows)
from services.robotwin.success_metadata import fresh_physical_state, initialize_success_metadata
from services.robotwin import expert_terminal_evidence as terminal
from scripts import validate_robotwin_constraint_acceptance as legacy

KIND = 'command_index_eef_baseline_qualification_dev_v1'
TASK = 'place_dual_shoes'
FLAGS = {k:False for k in ('formal_evidence_eligible','confidence_eligible','acceptance_ready',
                          'publication_allowed','training_allowed','task_coverage_qualified')}

def json_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def contained(relative):
    p=(PACKAGE/relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(PACKAGE) or p==PACKAGE:
        raise ValueError('Contained package path required')
    return p

def source_audit(design):
    ref=read(contained(design['reference']));index=read(contained(design['source_index']))
    root=contained(design['source_index']).parent
    expected={'captured_reference_not_for_training.json','expert_attempt.json','expert_final_physical_state.json',
              'expert_terminal_evidence.json','observation_manifest.json','run.json'}
    if set(index['files'])!=expected:raise ValueError('Original six-artifact index changed')
    for name,h in index['files'].items():
        if digest(root/name)!=h:raise ValueError('Original evidence drift: '+name)
    attempt=read(contained(design['source_attempt']))
    source_manifest=read(contained(design['source_observation_manifest']))
    if (source_manifest['kind']!='fixed_failure_expert_observation_dev_v1'
        or source_manifest['design']!=read(PACKAGE/'source_design/design.json')
        or source_manifest['design_sha256']!=digest(PACKAGE/'source_design/design.json')):
        raise ValueError('Successful reference source manifest differs')
    if any(source_manifest[k] is not False for k in FLAGS if k in source_manifest):
        raise ValueError('Source observation must retain nonpublication identity')
    if (attempt['task']!=TASK or attempt['seed']!=10000 or attempt['status']!='reference_captured'
        or attempt['native_success']is not True or attempt['planner_success']is not True
        or attempt.get('cleanup_error') or attempt.get('terminal_observation_error')):
        raise ValueError('Source is not a fully captured successful expert observation')
    if samples_to_trajectory(attempt['samples'])!=ref['trajectory']:
        raise ValueError('Captured sequence differs from full original samples')
    if (ref['initial_scene']!=attempt['initial_scene'] or scene_fingerprint(ref['initial_scene'])!=design['expected_scene_sha256']
        or ref['provenance']['scene_sha256']!=design['expected_scene_sha256']
        or ref['provenance']['task']!=TASK or ref['provenance']['seed']!=10000
        or ref['provenance']['recording_method']!='native_expert_save_freq_trace_no_images'
        or ref['provenance']['source_episode_sha256']!=json_digest(dict(scene=ref['initial_scene'],trajectory=ref['trajectory']))):
        raise ValueError('Source scene/action identity mismatch')
    if ref['bundle']['action_contract']!={'position':'absolute','rotation':'absolute','quaternion_order':'wxyz',
        'gripper':'absolute_open_fraction','frame':'world'} or ref['replay_adapter_applied']is not False:
        raise ValueError('Only original absolute world EEF source supported')
    if ref['bundle']['provenance']['source']!='acceptance_reference_not_for_training':
        raise ValueError('Do not remove source not-for-training identity')
    ar=arrays(ref['trajectory'])
    if len(ar['left']['pose'])!=246 or len(command_rows(ref['trajectory']))!=246:
        raise ValueError('The complete frozen 246-frame sequence is required')
    compiled,audit=legacy.reference_bundle(legacy.load_template(contained(design['template'])),ref['trajectory'],10000,design['expected_scene_sha256'])
    if not derived_values_equal(json.loads(json.dumps(compiled)),ref['bundle']) or not derived_values_equal(json.loads(json.dumps(audit)),ref['template_audit']):
        raise ValueError('Reference bundle does not reproduce frozen source compilation')
    terminal.validate_receipt(read(root/'expert_terminal_evidence.json'),read(root/'expert_final_physical_state.json'),
        frozen_policy=design['terminal_policy'],attempt=attempt)
    score=trajectory_violations(ref['bundle'],ref['trajectory'],ref['trajectory'])
    if not score['passed']:raise ValueError('Frozen reference rejects itself')
    return ref,dict(source_index_sha256=digest(root/'artifact_index.json'),reference_sha256=digest(contained(design['reference'])),
        trajectory_sha256=json_digest(ref['trajectory']),frames=246,commanded_actions=245,
        self_constraint_score=score,physics_timestamps_present=False,expert_full_initial_state_present=False,
        grippers={arm:dict(min=float(row['gripper'].min()),max=float(row['gripper'].max()),
            quaternion_norm_max_error=float(np.abs(np.linalg.norm(row['pose'][:,3:],axis=-1)-1).max())) for arm,row in ar.items()})

def validate_design(design, *, external=True, numeric=True):
    fixed=dict(version=1,kind=KIND,task=TASK,seed=10000,split='dev',new_experts=0,baseline_replays_max=2,
        variants=0,automatic_retries=0,replace_failed_seed=False,second_baseline_requires_first_reliable=True,
        command_frames=246,commanded_actions_per_replay=245,physics_time_preserved=False,
        source_trajectory_label_unchanged=True,initial_full_physical_state_equality_proven=False,
        prior_failures_immutable=True,source_commit='19620d792353ac737b3b46488cdc284a95a8a8af',
        alignment=ALIGNMENT,native_success_rule=SUCCESS_RULE,**FLAGS)
    for key,value in fixed.items():
        if type(design.get(key))is not type(value)or design.get(key)!=value:raise ValueError('Fixed contract changed: '+key)
    for rel,h in design['input_files'].items():
        if digest(contained(rel))!=h:raise ValueError('Input changed: '+rel)
    for rel,h in design['rax_files'].items():
        if digest(contained('code/rax/'+rel))!=h:raise ValueError('Frozen original code changed: '+rel)
    if external:
        for path,h in design['external_files'].items():
            if not Path(path).is_absolute()or digest(path)!=h:raise ValueError('Native/source evidence changed: '+path)
    if numeric:
        import sapien
        if design['runtime_numeric_context']!={'numpy_version':np.__version__,'sapien_version':sapien.__version__}:
            raise ValueError('Production numeric context changed')
    excluded=read(contained(design['training_exclusions']));exposure=read(contained(design['exposure_inventory']))
    if (10000 in excluded['excluded_training_seeds'] or design['expected_scene_sha256']in excluded['excluded_training_scene_sha256']
        or exposure['task']!=TASK or 10000 not in exposure['exposed_development_seeds'] or 10000 in exposure['reserved_holdout_seeds']):
        raise ValueError('Exposed-development/training/holdout identity changed')
    return source_audit(design)

def check_initial(native,reference):
    scene=initial_scene(native);first=sample(native)
    if scene!=reference['initial_scene']:raise ValueError('Exact available initial scene differs; no action executed')
    for arm in ('left','right'):
        expected=np.asarray(reference['trajectory'][arm]['pose'][0]);current=np.asarray(first[arm]['pose'])
        if (not np.allclose(current[:3],expected[:3],rtol=0,atol=1e-5)
            or abs(float(current[3:]@expected[3:]))<1-1e-6
            or abs(first[arm]['gripper']-reference['trajectory'][arm]['gripper'][0])>1e-5):
            raise ValueError('Initial EEF/gripper differs from source anchor')
    physical=fresh_physical_state(native)
    return dict(scene=scene,first_eef=first,physical=physical,
                source_full_physical_comparison_available=False,exact_available_scene_match=True)

def terminal_observation(native,raw,policy):
    # Reuse audited arithmetic/role binding, with a truthful new replay boundary.
    terminal.validate_policy(TASK,policy)
    if terminal.policy(TASK)!=policy:raise ValueError('Terminal numeric policy changed')
    terminal._audit_native(native)
    before=fresh_physical_state(native)
    roles={role:terminal._pose(role,getattr(native,role),before)for role in terminal.ROLES}
    grip={side:float(getattr(native.robot,side+'_gripper_val'))for side in ('left','right')}
    result=terminal.predicate_components(roles,grip,policy['numpy_version'])
    after=fresh_physical_state(native)
    if before!=after:raise ValueError('Terminal getters changed physical state')
    if result['conjunction']is not raw['final_success']:raise ValueError('Final unchanged native predicate/components differ')
    return dict(boundary='after_existing_replay_final_check_before_cleanup',physical=before,
                physical_after_sha256=json_digest(after),named_roles=roles,gripper_operands=grip,
                component_result=result,predicate_calls_added=0,physics_steps_added=0,controls_added=0)

def derive_trial(raw,trajectory,observation,reference,design):
    if trajectory['commanded']!=reference['trajectory'] or trajectory['alignment']!=ALIGNMENT:
        raise ValueError('Commanded sequence changed, inserted hold, truncated, or resampled')
    if raw['trial_id']not in ('baseline_0','baseline_1')or raw['seed']!=10000 or raw['task']!=TASK:
        raise ValueError('Wrong fixed baseline identity')
    command_score=trajectory_violations(reference['bundle'],reference['trajectory'],trajectory['commanded'])
    actual_score=None;track=None
    complete=raw.get('execution_complete')is True
    if complete:
        if raw.get('replay_started')is not True:raise ValueError('Complete replay requires replay_started')
        if (raw.get('alignment')!=ALIGNMENT or raw.get('native_success_rule')!=SUCCESS_RULE
            or raw.get('early_termination_suppressed_for_complete_alignment')is not True
            or raw.get('error') or raw.get('replay_adapter_applied')is not True
            or not derived_values_equal(raw.get('replay_adapter'),reference['replay_adapter'])):
            raise ValueError('Complete replay protocol/adapter metadata inconsistent')
        arrays(trajectory['actual'])
        if len(trajectory['actual']['left']['pose'])!=246:raise ValueError('Incomplete aligned replay')
        track=tracking_report(trajectory['actual'],trajectory['commanded'])
        actual_score=trajectory_violations(reference['bundle'],reference['trajectory'],trajectory['actual'])
    if (raw.get('native_success') is not raw.get('ever_success') or
        (raw.get('final_success')is True and raw.get('ever_success')is not True)):
        raise ValueError('Native ever/final evidence inconsistent')
    statuses=raw.get('planner_results',[])
    expected_plans=sorted((i,arm)for i in range(1,246)for arm in ('left','right'))
    planner_ok=(len(statuses)==490 and sorted((row['frame'],row['arm'])for row in statuses)==expected_plans
                and all(row['status']=='Success'for row in statuses))
    if raw.get('planner_success')is not planner_ok:raise ValueError('Planner statuses do not reproduce eligibility')
    terminal_ok=False
    if observation.get('terminal'):
        obs=observation['terminal'];physical=obs['physical'];roles=obs['named_roles']
        if (obs['boundary']!='after_existing_replay_final_check_before_cleanup'
            or obs['physical_after_sha256']!=json_digest(physical)
            or any(obs[k]!=0 for k in ('predicate_calls_added','physics_steps_added','controls_added'))
            or set(roles)!=set(terminal.ROLES) or obs['gripper_operands']!=physical['gripper_commands']):
            raise ValueError('Terminal observation identity changed')
        indices=[]
        for role,row in roles.items():
            i=row['physical_actor_index'];indices.append(i)
            if type(i)is not int or i<0 or i>=len(physical['actors']):raise ValueError('Terminal actor index invalid')
            if row['name']!=terminal.ROLES[role]or any(row[k]!=physical['actors'][i][k]for k in ('name','p','q','position_dtype','quaternion_dtype')):
                raise ValueError('Terminal role is not physical actor')
        if len(set(indices))!=3:raise ValueError('Terminal actors alias')
        component=terminal.predicate_components(roles,obs['gripper_operands'],design['terminal_policy']['numpy_version'])
        if component!=obs['component_result']or component['conjunction']is not raw.get('final_success'):
            raise ValueError('Terminal predicate reconstruction changed')
        terminal_ok=True
    initial=observation.get('initial');initial_ok=False
    if initial:
        if (initial['scene']!=reference['initial_scene'] or initial['exact_available_scene_match']is not True
            or initial['source_full_physical_comparison_available']is not False):raise ValueError('Initial scene evidence changed')
        for arm in ('left','right'):
            exp=np.asarray(reference['trajectory'][arm]['pose'][0]);cur=np.asarray(initial['first_eef'][arm]['pose'])
            if (not np.allclose(cur[:3],exp[:3],rtol=0,atol=1e-5)or abs(float(cur[3:]@exp[3:]))<1-1e-6
                or abs(initial['first_eef'][arm]['gripper']-reference['trajectory'][arm]['gripper'][0])>1e-5):raise ValueError('Initial anchor receipt differs')
        if complete and any(trajectory['actual'][arm]['pose'][0]!=initial['first_eef'][arm]['pose']
            or trajectory['actual'][arm]['gripper'][0]!=initial['first_eef'][arm]['gripper']for arm in ('left','right')):
            raise ValueError('Replay sample0 and initial anchor receipt differ')
        initial_ok=True
    errors=raw.get('errors',[])
    eligible=complete and planner_ok and track is not None and track['realized']and not errors and initial_ok and terminal_ok
    reliable=eligible and raw.get('native_success')is True and actual_score['passed']and command_score['passed']
    return dict(trial_id=raw['trial_id'],kind='baseline',execution_complete=complete,planner_success=planner_ok,
        realized=None if track is None else track['realized'],tracking=track,native_success=raw.get('native_success'),
        ever_success=raw.get('ever_success'),final_success=raw.get('final_success'),eligible=bool(eligible),
        commanded_constraint_pass=command_score['passed'],actual_constraint_pass=None if actual_score is None else actual_score['passed'],
        commanded_score=command_score,actual_score=actual_score,baseline_reliable=bool(reliable),
        terminal_evidence_complete=terminal_ok,initial_evidence_complete=initial_ok,replay_started=raw.get('replay_started')is True,errors=errors)

def run_trial(output,trial_id,reference,design,factory):
    directory=output/'trials'/trial_id;directory.mkdir(parents=True,exist_ok=False)
    atomic_json(directory/'intent.json',dict(kind=KIND,trial_id=trial_id,seed=10000,new_experts=0,command_sha256=json_digest(reference['trajectory'])))
    raw=dict(task=TASK,seed=10000,trial_id=trial_id,native_success=None,ever_success=None,final_success=None,
        planner_success=False,execution_complete=False,planner_results=[],errors=[],replay_started=False)
    trajectory=dict(commanded=reference['trajectory'],actual={'left':{'pose':[],'gripper':[]},'right':{'pose':[],'gripper':[]}},alignment=ALIGNMENT)
    observations={};native=None;started=time.monotonic()
    try:
        native=factory(10000)
        observations['initial']=check_initial(native,reference)
        adapter=install_base_frame_adapter(native)
        if not derived_values_equal(adapter,reference['replay_adapter']):raise ValueError('Original fixed SE3 adapter changed')
        raw['replay_adapter']=adapter;raw['replay_adapter_applied']=True
        raw['replay_started']=True
        result=replay(native,reference['trajectory']);trajectory['actual']=result.pop('actual');raw.update(result)
        observations['terminal']=terminal_observation(native,raw,design['terminal_policy'])
    except BaseException as exc:
        raw['errors'].append(dict(phase='execution_or_observation',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        if getattr(exc,'_qualification_setup_cleanup_error',None) is not None:
            raw['errors'].append(dict(phase='setup_cleanup',**exc._qualification_setup_cleanup_error))
    finally:
        if native is not None:
            try:native.close_env()
            except BaseException as exc:raw['errors'].append(dict(phase='cleanup',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
    raw['seconds']=time.monotonic()-started
    for name,value in [('outcome.json',raw),('trajectory.json',trajectory),('observations.json',observations)]:atomic_json(directory/name,value)
    record=dict(trial_id=trial_id,files={name:digest(directory/name)for name in ('intent.json','outcome.json','trajectory.json','observations.json')},
        derived=derive_trial(raw,trajectory,observations,reference,design))
    atomic_json(directory/'record.json',record)
    return record['derived']

def expected_manifest(design):
    return dict(version=1,kind=KIND,design_sha256=json_digest(design),reference_sha256=digest(contained(design['reference'])),
        source_index_sha256=digest(contained(design['source_index'])),task=TASK,seed=10000,
        trial_plan=['baseline_0','baseline_1'],second_baseline_requires_first_reliable=True,tracking_limits=TRACKING,
        new_experts=0,variants=0,automatic_retries=0,physics_time_preserved=False,**FLAGS)

def verify_output(output,design):
    reference,_=source_audit(design);manifest=read(output/'manifest.json')
    if manifest!=expected_manifest(design):raise ValueError('Baseline-only manifest identity or budget changed')
    state=read(output/'run.json');index=read(output/'artifact_index.json')
    files={str(p.relative_to(output)):digest(p)for p in sorted(output.rglob('*'))if p.is_file()and p.name!='artifact_index.json'and 'native_scratch'not in p.parts}
    if index!={'kind':KIND,'files':files}:raise ValueError('Output index incomplete or tampered')
    rows=[];directories=sorted((output/'trials').iterdir())if(output/'trials').exists()else[]
    if [p.name for p in directories]not in (['baseline_0'],['baseline_0','baseline_1']):raise ValueError('Fixed conditional prefix required')
    for directory in directories:
        record=read(directory/'record.json')
        if set(p.name for p in directory.iterdir())!={'intent.json','outcome.json','trajectory.json','observations.json','record.json'}:
            raise ValueError('Unexpected or partial trial artifacts')
        required={'intent.json','outcome.json','trajectory.json','observations.json'}
        if set(record['files'])!=required or any(digest(directory/n)!=h for n,h in record['files'].items()):raise ValueError('Trial raw hashes differ')
        intent=read(directory/'intent.json')
        if intent!=dict(kind=KIND,trial_id=directory.name,seed=10000,new_experts=0,command_sha256=json_digest(reference['trajectory'])):raise ValueError('Trial intent changed')
        row=derive_trial(read(directory/'outcome.json'),read(directory/'trajectory.json'),read(directory/'observations.json'),reference,design)
        if row['trial_id']!=directory.name or record['trial_id']!=directory.name or not derived_values_equal(row,record['derived']):raise ValueError('Derived trial evidence differs')
        if rows and not rows[0]['baseline_reliable']:raise ValueError('Second baseline ran after unreliable first')
        rows.append(row)
    qualified=len(rows)==2 and all(row['baseline_reliable']for row in rows)
    expected='command_sequence_baselines_qualified'if qualified else'blocked_baseline_qualification'
    if state!={'kind':KIND,'status':expected,'completed_trials':len(rows),'baseline_reliable':qualified,**FLAGS}:
        raise ValueError('Final qualification state differs from fixed evidence')
    if len(rows)==1 and rows[0]['baseline_reliable']:raise ValueError('Reliable first baseline requires second')
    return dict(kind=KIND,status=expected,rows=rows,baseline_reliable=qualified,new_experts=0,
        baseline_attempts=len(rows),native_initialization_attempts=len(rows),baseline_replays=sum(row['replay_started']for row in rows),variants=0,automatic_retries=0,physics_time_preserved=False,
        initial_full_physical_state_equality_proven=False,manifest_sha256=digest(output/'manifest.json'),
        artifact_index_sha256=digest(output/'artifact_index.json'),**FLAGS)

def preserve_setup_cleanup_error(native,primary):
    try:native.close_env()
    except BaseException as cleanup:
        primary._qualification_setup_cleanup_error=dict(type=type(cleanup).__name__,message=str(cleanup),traceback=traceback.format_exc())


def run(args):
    design=read(args.design);reference,audit=validate_design(design)
    if args.check_only:
        print(json.dumps(dict(kind=KIND,source_audit=audit,gpu_started=False,**FLAGS)));return 0
    if not args.execute:raise ValueError('Explicit execute required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'manifest.json',expected_manifest(design))
    from services.robotwin.runtime import initialize_runtime,native_config
    from services.robotwin.collection import install_planner_cache
    os.environ['ROBOTWIN_PATH']=design['native_root'];initialize_runtime();install_planner_cache()
    cls=getattr(importlib.import_module('envs.'+TASK),TASK);kwargs=native_config(TASK,0)
    kwargs.update(need_plan=True,save_data=False,render_freq=0,eval_video_log=False,eval_video_save_dir=None,save_path=str(output/'native_scratch'))
    def factory(seed):
        random.seed(seed);native=cls()
        try:
            native.setup_demo(now_ep_num=0,seed=seed,is_test=False,**kwargs)
            initialize_success_metadata(TASK,native,expected_scene_sha256=scene_fingerprint(initial_scene(native)))
        except BaseException as primary:
            preserve_setup_cleanup_error(native,primary)
            raise
        return native
    execute_sequence(output,reference,design,factory)
    return 0

def execute_sequence(output,reference,design,factory):
    rows=[]
    for trial_id in ('baseline_0','baseline_1'):
        row=run_trial(output,trial_id,reference,design,factory);rows.append(row)
        if not row['baseline_reliable']:break
    qualified=len(rows)==2 and all(row['baseline_reliable']for row in rows)
    atomic_json(output/'run.json',dict(kind=KIND,status='command_sequence_baselines_qualified'if qualified else'blocked_baseline_qualification',completed_trials=len(rows),baseline_reliable=qualified,**FLAGS))
    atomic_json(output/'artifact_index.json',dict(kind=KIND,files={str(p.relative_to(output)):digest(p)for p in sorted(output.rglob('*'))if p.is_file()and p.name!='artifact_index.json'and 'native_scratch'not in p.parts}))
    return verify_output(output,design)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--check-only',action='store_true');parser.add_argument('--execute',action='store_true')
    raise SystemExit(run(parser.parse_args()))
