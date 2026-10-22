#!/usr/bin/env python3
"""Record N successful, distinct clean initial states; publish after RAX annotation.

No randomized scenes are selected. Successful native plan/replay recordings go
to raw/ with an annotation queue entry. data/ contains only reviewed recordings.
"""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.robotwin.collection import atomic_json, initial_scene, scene_fingerprint, install_planner_cache
from services.robotwin.runtime import initialize_runtime,native_config,source_commit
from services.robotwin.native_compat import compatibility_report
from scripts.collect_robotwin_expert import publish_scene_info


class RejectedEpisode(Exception):pass


def play_expert(native):
    """Keep native fallback logic, but classify propagated infeasible grasp poses."""
    original=native.get_grasp_pose;missing=False
    def tracked(*args,**kwargs):
        nonlocal missing
        result=original(*args,**kwargs)
        missing=missing or result is None
        return result
    native.get_grasp_pose=tracked
    try:return native.play_once()
    except (TypeError,AssertionError) as exc:
        if missing and str(exc) in ("'NoneType' object is not subscriptable",'target_pose cannot be None for move action.'):
            raise RejectedEpisode('native_grasp_pose_unavailable: '+str(exc)) from exc
        raise
    finally:native.get_grasp_pose=original


def inventory(output):
    import h5py
    entries={}
    for directory in ('raw','data'):
        for path in (output/directory).glob('episode*.hdf5'):
            index=int(path.stem.removeprefix('episode'))
            with h5py.File(path,'r') as h:
                meta=json.loads(h.attrs['robotwin_expert_provenance'])
            if index in entries:raise ValueError('Duplicate raw/published episode')
            entries[index]=dict(meta,index=index,path=str(path))
    ordered=[entries[i] for i in sorted(entries)]
    if sorted(entries)!=list(range(len(entries))):raise ValueError('Noncontiguous episode inventory')
    if len({e['seed'] for e in ordered})!=len(ordered):raise ValueError('Duplicate clean seed')
    if len({e['initial_scene_sha256'] for e in ordered})!=len(ordered):raise ValueError('Duplicate initial geometry')
    return ordered


def record(native,task,seed,index,stage,output,used_scenes,revision,cache_stats):
    import h5py
    from generate_episode_instructions import generate_episode_descriptions
    from services.robotwin.constraint_schema import HDF5_KEY,canonical_json
    stage.mkdir(parents=True,exist_ok=False)
    args=native_config(task,0)
    args.update(save_path=str(stage),need_plan=True,save_data=False,render_freq=0,eval_video_log=False)
    started=time.time();initial=None
    try:
        random.seed(seed)
        native.setup_demo(now_ep_num=index,seed=seed,is_test=False,**args)
        initial=initial_scene(native);fingerprint=scene_fingerprint(initial)
        atomic_json(stage/'initial_scene.json',initial)
        if fingerprint in used_scenes:raise RejectedEpisode('duplicate_initial_geometry')
        info=play_expert(native)
        if not native.plan_success or not native.check_success():raise RejectedEpisode('expert_plan_failed')
        native.save_traj_data(index)
        random.seed(seed)
        choices=generate_episode_descriptions(task,[info['info']],1)[0]['seen']
        instruction=random.Random(seed).choice(choices)
    finally:native.close_env()
    planned=time.time()
    args.update(need_plan=False,save_data=True)
    try:
        random.seed(seed)
        native.setup_demo(now_ep_num=index,seed=seed,is_test=False,**args)
        replay_initial=initial_scene(native)
        if scene_fingerprint(replay_initial)!=fingerprint:raise RejectedEpisode('plan_replay_initial_state_mismatch')
        native.set_instruction(instruction)
        native.set_path_lst(dict(args,**native.load_tran_data(index)))
        info=play_expert(native)
        if not native.plan_success or not native.check_success():raise RejectedEpisode('recording_replay_failed')
    finally:native.close_env()
    replayed=time.time()
    native.merge_pkl_to_hdf5_video();native.remove_data_cache()
    source=stage/'data'/f'episode{index}.hdf5'
    meta={'task':task,'seed':seed,'index':index,'collection':0,'task_config':'demo_clean',
          'source_commit':revision,'native_success':True,'initial_scene_sha256':fingerprint,
          'recording_method':'native_plan_replay_merge_pkl_to_hdf5_video','save_freq':args['save_freq'],
          'runtime_compatibility':compatibility_report(),
          'planner_cache':dict(cache_stats),'seconds':{'plan':planned-started,'replay':replayed-planned,'merge':time.time()-replayed}}
    bundle={'schema_version':'1.0','action_contract':{'position':'absolute','rotation':'absolute',
            'quaternion_order':'wxyz','gripper':'absolute_open_fraction','frame':'world'},
            'provenance':{'split':'train','variant':'clean_50','recording_id':f'{task}-clean-seed{seed}-{fingerprint[:12]}',
                'source':'robotwin_native_expert_rax_annotation','native_success':True,
                'sampling':f'robotwin_native_save_freq_{args["save_freq"]}'},'constraints':[]}
    with h5py.File(source,'r+') as h:
        h.attrs['instruction']=instruction
        h.attrs['robotwin_expert_provenance']=json.dumps(meta)
        h.create_dataset('initial_scene/json',data=json.dumps(initial),dtype=h5py.string_dtype('utf-8'))
        h.create_dataset(HDF5_KEY,data=canonical_json(bundle),dtype=h5py.string_dtype('utf-8'))
        h.create_dataset('trajectory_constraints/history',data='[]',dtype=h5py.string_dtype('utf-8'))
        h['trajectory_constraints'].attrs['revision']=0
        frames=len(h['endpose/left_endpose'])
        if frames<2:raise RejectedEpisode('recording_too_short')
        for arm in ('left','right'):
            if len(h[f'endpose/{arm}_endpose'])!=frames:raise RejectedEpisode('inconsistent_frame_count')
    target=output/'raw'/source.name;target.parent.mkdir(exist_ok=True)
    if target.exists() or (output/'data'/source.name).exists():raise FileExistsError(target)
    publish_scene_info(output,index,info)
    source.rename(target)
    receipt=dict(meta,frames=frames,path=str(target),annotation_status='pending')
    atomic_json(output/'annotation_queue'/f'episode{index}.json',receipt)
    atomic_json(stage/'receipt.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=50);p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--max-attempts',type=int,default=1000)
    args=p.parse_args()
    if not 0<args.episodes<=50 or args.seed_start<0 or args.max_attempts<args.episodes:p.error('Invalid collection limits')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'.collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    revision=source_commit();entries=inventory(output)
    state_path=output/'collection_status.json'
    previous=json.loads(state_path.read_text()) if state_path.exists() else {}
    if previous and (previous['source_commit']!=revision or previous['target']!=args.episodes):raise ValueError('Resume identity/target mismatch')
    seed=max(args.seed_start,previous.get('next_seed',args.seed_start),max([e['seed']+1 for e in entries],default=0))
    state={'task':args.task,'target':args.episodes,'collection':'clean','source_commit':revision,
           'status':'initializing','completed_recordings':len(entries),'next_seed':seed,
           'unique_seeds':len(entries),'unique_initial_scenes':len(entries),'started_unix':previous.get('started_unix',time.time()),
           'annotation_completion':'Count accepted files in data/ separately; raw/ is pending RAX review.'}
    atomic_json(state_path,state)
    native=None
    try:
        initialize_runtime();cache=install_planner_cache()
        native=getattr(importlib.import_module('envs.'+args.task),args.task)()
        while len(entries)<args.episodes:
            if seed>=args.seed_start+args.max_attempts:raise RuntimeError('Candidate seed limit reached before quota')
            state.update(status='recording',active_seed=seed,next_seed=seed+1,updated_unix=time.time())
            atomic_json(state_path,state)
            stage=output/'attempts'/f'seed{seed}'
            try:
                receipt=record(native,args.task,seed,len(entries),stage,output,
                               {e['initial_scene_sha256'] for e in entries},revision,cache)
                entries.append(receipt)
                state.update(completed_recordings=len(entries),unique_seeds=len({e['seed'] for e in entries}),
                             unique_initial_scenes=len({e['initial_scene_sha256'] for e in entries}),last_receipt=receipt)
                atomic_json(output/'clean_manifest.json',{'task':args.task,'target':args.episodes,'entries':entries})
                print(json.dumps({'event':'recorded','task':args.task,'count':len(entries),'seed':seed,'frames':receipt['frames'],'seconds':receipt['seconds']}),flush=True)
            except Exception as exc:
                # Expected planning/unstable-scene rejection advances to a new seed.
                if not isinstance(exc,RejectedEpisode) and type(exc).__name__!='UnStableError':raise
                failure={'seed':seed,'error':f'{type(exc).__name__}: {exc}'}
                atomic_json(stage/'failure.json',failure)
                with (output/'failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
                shutil.rmtree(stage/'.cache',ignore_errors=True)
                print(json.dumps({'event':'rejected',**failure}),flush=True)
            seed+=1
        state.update(status='recorded',completed_recordings=len(entries))
    except BaseException as exc:
        state.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if native is not None:native.close_env()
        state['updated_unix']=time.time();atomic_json(state_path,state)


if __name__=='__main__':main()
