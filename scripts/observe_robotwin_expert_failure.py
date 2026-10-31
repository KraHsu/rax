#!/usr/bin/env python3
"""One fixed expert observation; never replay, retry, publish or train."""
import argparse
import importlib
import json
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from services.robotwin.collection import atomic_json,initial_scene,scene_fingerprint  # noqa: E402
from services.robotwin.success_metadata import initialize_success_metadata,metadata_policy  # noqa: E402
from services.robotwin import expert_terminal_evidence as shoes  # noqa: E402
from services.robotwin import cabinet_terminal_evidence as cabinet  # noqa: E402
from services.robotwin.expert_planner_observer import ExpertPlannerObserver,REQUESTS,returned_status,policy as planner_policy  # noqa: E402
from services.robotwin.control_timing_observer import file_hash  # noqa: E402
from scripts import validate_robotwin_constraint_acceptance as legacy  # noqa: E402

KIND='fixed_failure_expert_observation_dev_v1'
TASKS={'place_dual_shoes':dict(seed=10000,scene='6206f46a75953222458bfdfc64f4df663fcfd6c2017e317cb10b736518fc3976'),
       'put_object_cabinet':dict(seed=10050,scene='710aca71c27f37f5dce7d3daea0d3023998208391f8a7ce78390a6fee0da24d5')}
FLAGS=dict(formal_evidence_eligible=False,confidence_eligible=False,acceptance_ready=False,publication_allowed=False,training_allowed=False)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def within(root,relative):
    p=(root/relative).resolve()
    if Path(relative).is_absolute()or not p.is_relative_to(root.resolve())or p==root.resolve():raise ValueError('Inputs must be contained relative files')
    return p


def validate_design(root,design):
    task=design.get('task')
    if (design.get('version')!=1 or design.get('kind')!=KIND or task not in TASKS
            or design.get('seed')!=TASKS[task]['seed']or design.get('expected_scene_sha256')!=TASKS[task]['scene']):raise ValueError('Wrong fixed task/seed/failed scene')
    for k,v in dict(split='dev',expert_attempts_max=1,legacy_replays=0,candidate_variants=0,automatic_retries=0,replace_failed_seed=False,observational_reproduction=True).items():
        if design.get(k)!=v:raise ValueError('Finite failure-observation contract changed: '+k)
    for k,v in FLAGS.items():
        if design.get(k)is not v:raise ValueError('Observation cannot authorize formal evidence')
    for rel,h in design['input_files'].items():
        if file_hash(within(root,rel))!=h:raise ValueError('Input drift: '+rel)
    for rel,h in design['rax_files'].items():
        if file_hash(within(ROOT,rel))!=h:raise ValueError('Code drift: '+rel)
    for path,h in design['external_files'].items():
        if not Path(path).is_absolute()or file_hash(path)!=h:raise ValueError('External source/history drift: '+path)
    native_root=Path(design['native_root'])
    native_required={str(native_root/p)for p in ('envs/'+task+'.py','envs/_base_task.py','envs/robot/robot.py','envs/robot/planner.py','envs/utils/actor_utils.py','envs/utils/action.py','envs/utils/create_actor.py')}
    if not native_required<=set(design['external_files']):raise ValueError('Native control/predicate sources omitted')
    required={'scripts/observe_robotwin_expert_failure.py','scripts/validate_robotwin_constraint_acceptance.py',
              'scripts/collect_robotwin_clean50.py','services/robotwin/collection.py','services/robotwin/runtime.py',
              'services/robotwin/success_metadata.py','services/robotwin/instruction_metadata.py','services/robotwin/expert_terminal_evidence.py',
              'services/robotwin/cabinet_terminal_evidence.py','services/robotwin/expert_planner_observer.py'}
    if not required<=set(design['rax_files']):raise ValueError('Mandatory driver/observer/factory source omitted')
    excluded=read(within(root,design['training_exclusions']))
    if design['seed']in excluded['excluded_training_seeds']or design['expected_scene_sha256']in excluded['excluded_training_scene_sha256']:
        raise ValueError('Fixed failure scene conflicts with training exclusions')
    exposure=read(within(root,design['exposure_inventory']))
    if (exposure.get('task')!=task or design['seed']not in exposure.get('exposed_development_seeds',[])
            or design['seed']in exposure.get('reserved_holdout_seeds',[])):raise ValueError('Previously exposed dev/holdout identity changed')
    prior=read(within(root,design['previous_failure']))
    if prior['seed']!=design['seed']or prior['scene_sha256']!=design['expected_scene_sha256']or prior['status']!='failed' or prior.get('planner_success')is not(task=='place_dual_shoes') or prior.get('native_success')is not(False if task=='place_dual_shoes'else None):raise ValueError('Original failed attempt must be retained')
    history=read(within(root,design['initialization_history']))
    if task=='place_dual_shoes':
        if (history['previous_stagea_initializations']!=5 or history['new_stagea_initializations_max']!=1 or history['cumulative_stagea_initializations_max']!=6):raise ValueError('StageA history budget changed')
    elif history['previous_cabinet_smoke_initializations']!=1 or history['new_cabinet_initializations_max']!=1 or history['cumulative_cabinet_initializations_max']!=2:
        raise ValueError('Cabinet failure-observation budget changed')
    template=legacy.load_template(within(root,design['template']))
    if template['task']!=task:raise ValueError('Wrong task template')
    expected=design['runtime_numeric_context']
    import sapien
    if expected!={'numpy_version':np.__version__,'sapien_version':sapien.__version__}:raise ValueError('Production numeric context changed')
    terminal_policy=shoes.policy(task)if task=='place_dual_shoes'else cabinet.policy(within(root,design['cabinet_asset_metadata']))
    if terminal_policy!=design['terminal_policy']or metadata_policy(task)!=design['success_metadata_policy']:
        raise ValueError('Frozen terminal/fresh metadata policy changed')
    if design['planner_policy']!=(planner_policy()if task=='put_object_cabinet'else None):raise ValueError('Planner observation policy changed')
    return template,excluded


def commit_index(output):
    files={str(p.relative_to(output)):file_hash(p)for p in sorted(output.rglob('*'))if p.is_file()and p.name!='artifact_index.json'and 'native_scratch'not in p.parts}
    atomic_json(output/'artifact_index.json',dict(kind=KIND,files=files))


def run_one(output,design,template,excluded,base_factory):
    """Exactly one legacy expert capture; every exit retains original evidence."""
    task=design['task'];seed=design['seed'];observer=None;calls=0
    run=dict(kind=KIND,task=task,seed=seed,status='observation_started',native_initializations_started=0,
             expert_attempts_started=0,legacy_replays=0,automatic_retries=0,**FLAGS)
    atomic_json(output/'run.json',run)
    def factory(fixed_seed):
        nonlocal calls,observer
        if calls or fixed_seed!=seed:raise ValueError('No additional factory/seed permitted')
        calls+=1;run['native_initializations_started']=1;atomic_json(output/'run.json',run)
        native=base_factory(fixed_seed)
        try:
            scene=initial_scene(native)
            if scene_fingerprint(scene)!=design['expected_scene_sha256']:raise ValueError('Initial scene differs from original failed scene')
            if task=='put_object_cabinet':observer=ExpertPlannerObserver(native,frozen_policy=design['planner_policy']).start()
        except BaseException:
            if observer is not None:observer.stop()
            native.close_env();raise
        run['expert_attempts_started']=1;atomic_json(output/'run.json',run)
        return native
    def terminal(native,attempt):
        errors=[]
        if observer is not None:
            try:observer.stop();atomic_json(output/'planner_observation.json',observer.receipt())
            except Exception as exc:errors.append('planner_observation: '+type(exc).__name__+': '+str(exc))
        try:
            module=shoes if task=='place_dual_shoes'else cabinet
            state,receipt=module.observe(native,attempt,frozen_policy=design['terminal_policy'])
            atomic_json(output/'expert_final_physical_state.json',state);atomic_json(output/'expert_terminal_evidence.json',receipt)
        except Exception as exc:errors.append('terminal_observation: '+type(exc).__name__+': '+str(exc))
        if errors:
            atomic_json(output/'observation_errors.json',dict(errors=errors));raise ValueError('; '.join(errors))
    primary=None
    try:
        reference=legacy.capture_reference(factory,template,seed,excluded['excluded_training_scene_sha256'],
            attempt_path=output/'expert_attempt.json',terminal_observer=terminal)
        atomic_json(output/'captured_reference_not_for_training.json',reference)
    except BaseException as exc:
        primary=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        atomic_json(output/'original_capture_error.json',primary)
    finally:
        if observer is not None:observer.stop()
        attempt=read(output/'expert_attempt.json')if(output/'expert_attempt.json').exists()else None
        complete=bool(attempt and attempt.get('terminal_observer_invoked')is True
            and not attempt.get('terminal_observation_error')and not attempt.get('cleanup_error')
            and(output/'expert_terminal_evidence.json').exists())
        if task=='put_object_cabinet':
            complete=complete and(output/'planner_observation.json').exists()
            if complete:complete=read(output/'planner_observation.json')['owner_thread_scope_complete']is True
        run.update(status='observation_complete'if complete else'observation_incomplete',
                   original_capture_error=primary,original_planner_success=None if attempt is None else attempt.get('planner_success'),
                   original_native_success=None if attempt is None else attempt.get('native_success'),native_result_replaced=False)
        atomic_json(output/'run.json',run);commit_index(output)
    return run


def validate_planner_receipt(receipt,frozen):
    if (receipt.get('version')!=frozen['version']or receipt.get('policy')!=frozen or receipt.get('formal_evidence_eligible')is not False
            or receipt.get('original_arguments_modified')is not False or receipt.get('control_methods_wrapped')is not False
            or receipt.get('all_process_or_solver_calls_observed')is not False
            or any(receipt.get(k)!=0 for k in ('extra_state_getters','extra_physics_steps','extra_get_info_calls','extra_predicate_calls'))):raise ValueError('Invalid planner observation contract')
    if len(receipt['events'])>frozen['max_events']:raise ValueError('Planner event budget exceeded')
    calls={};returned=set();first=None;fatal=None;last=None
    for row in receipt['events']:
        cid=row['call_id']
        if row['event']=='call':
            if type(cid)is not int or cid!=len(calls)+1:raise ValueError('Nonmonotonic/duplicate observed call')
            calls[cid]=row
        elif row['event']=='return_or_unwind':
            if cid not in calls or cid in returned:raise ValueError('Unpaired/duplicate return')
            returned.add(cid)
            entry=calls[cid];status=row['returned_status']
            raw_value=row['returned_cpu_value'];raw=raw_value.get('status')if type(raw_value)is dict else None
            if type(raw)is dict and raw.get('kind')=='numpy':raw=raw.get('values')
            expected_status=returned_status({'status':raw})if row['method']in REQUESTS else None
            if status!=expected_status:raise ValueError('Status differs from retained returned CPU value')
            if status is not None:
                if not status['all_success']:
                    last=cid
                    if first is None:first=cid
            if entry['method']!=row['method']or entry['source']!=row['source']:raise ValueError('Call/return identity mismatch')
            if entry['plan_success']is True and row['plan_success']is False and fatal is None:
                fatal=cid
                declared=receipt['first_true_to_false_scope']
                if declared is None or declared['last_recognized_non_success_request']!=last or declared['method']!=row['method']:raise ValueError('Fatal scope attribution changed')
        else:raise ValueError('Unknown observation event')
    if first!=receipt['first_non_success_request_id']or fatal!=(None if receipt['first_true_to_false_scope']is None else receipt['first_true_to_false_scope']['call_id']):raise ValueError('First-failure classification changed')
    if receipt['owner_thread_scope_complete']and returned!=set(calls):raise ValueError('Complete receipt has unfinished observed calls')
    if receipt['owner_thread_scope_complete']is not(not receipt['errors']):raise ValueError('Planner completeness differs from raw errors')


def verify_output(output,design_hash,expected_design):
    output=Path(output);index=read(output/'artifact_index.json');manifest=read(output/'observation_manifest.json');design=manifest['design']
    if index['kind']!=KIND or manifest['kind']!=KIND or manifest['design_sha256']!=design_hash or design!=expected_design:raise ValueError('Wrong observation manifest/index')
    present={str(p.relative_to(output))for p in output.rglob('*')if p.is_file()and p.name!='artifact_index.json'and 'native_scratch'not in p.parts}
    if present!=set(index['files']):raise ValueError('Unindexed/missing evidence')
    for rel,h in index['files'].items():
        if file_hash(within(output,rel))!=h:raise ValueError('Observation artifact changed')
    run=read(output/'run.json');task=design['task']
    for value in (manifest,run):
        if any(value.get(k)is not v for k,v in FLAGS.items()):raise ValueError('Observation cannot be formal evidence')
    if (run['kind']!=KIND or run['task']!=task or run['seed']!=design['seed']or run['native_initializations_started']not in (0,1)
            or run['expert_attempts_started']not in (0,1)or run['legacy_replays']!=0 or run['automatic_retries']!=0 or run['native_result_replaced']is not False):raise ValueError('Episode budget/identity changed')
    attempt=read(output/'expert_attempt.json')
    if attempt['task']!=task or attempt['seed']!=design['seed']:raise ValueError('Wrong original attempt')
    complete=False
    if attempt.get('terminal_observer_invoked')is True and not attempt.get('terminal_observation_error')and not attempt.get('cleanup_error'):
        if attempt['scene_sha256']!=design['expected_scene_sha256']:raise ValueError('Wrong failed-scene identity')
        module=shoes if task=='place_dual_shoes'else cabinet
        module.validate_receipt(read(output/'expert_terminal_evidence.json'),read(output/'expert_final_physical_state.json'),
            frozen_policy=design['terminal_policy'],attempt=attempt)
        complete=True
        if task=='put_object_cabinet':
            receipt=read(output/'planner_observation.json');validate_planner_receipt(receipt,design['planner_policy']);complete=receipt['owner_thread_scope_complete']is True
    expected='observation_complete'if complete else'observation_incomplete'
    if run['status']!=expected or run['original_native_success']!=attempt.get('native_success')or run['original_planner_success']!=attempt.get('planner_success'):
        raise ValueError('Derived completion/original labels changed')
    if run['original_capture_error'] is not None:
        if read(output/'original_capture_error.json')!=run['original_capture_error']:raise ValueError('Original exception artifact changed')
    return dict(run,artifact_index_sha256=file_hash(output/'artifact_index.json'))


def run(args):
    design=read(args.design);root=args.design.resolve().parent;template,excluded=validate_design(root,design)
    from services.robotwin.runtime import initialize_runtime,native_config,source_commit
    if source_commit()!=design['native_commit']or legacy.training_exclusions(Path(design['training_root']))!=excluded:raise ValueError('Native revision/training exclusion changed')
    if args.check_only:
        print(json.dumps(dict(task=design['task'],checked=True,native_initializations=0,expert_attempts_max=1,legacy_replays=0,**FLAGS)));return 0
    if not args.execute:raise ValueError('Explicit execute flag required')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'observation_manifest.json',dict(kind=KIND,version=1,design=design,design_sha256=file_hash(args.design),created_unix=time.time(),**FLAGS))
    def interrupted(signum,_frame):raise InterruptedError('Expert observation interrupted by signal '+str(signum))
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,interrupted)
    initialize_runtime()
    from services.robotwin.collection import install_planner_cache
    install_planner_cache();task=design['task'];cls=getattr(importlib.import_module('envs.'+task),task)
    kwargs=native_config(task,0);kwargs.update(need_plan=True,save_data=False,render_freq=0,eval_video_log=False,eval_video_save_dir=None,save_path=str(output/'native_scratch'))
    def factory(seed):
        random.seed(seed);native=cls()
        try:
            native.setup_demo(now_ep_num=0,seed=seed,is_test=False,**kwargs)
            initialize_success_metadata(task,native,expected_scene_sha256=scene_fingerprint(initial_scene(native)))
        except BaseException:native.close_env();raise
        return native
    result=run_one(output,design,template,excluded,factory);verify_output(output,file_hash(args.design),design);print(json.dumps(result))
    return 0 if result['status']=='observation_complete'else 2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--design',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--check-only',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args();raise SystemExit(run(a))

if __name__=='__main__':main()
