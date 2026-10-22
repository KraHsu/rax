#!/usr/bin/env python3
"""Persistent GPU work queue for all 50 native clean tasks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.robotwin.collection import atomic_json
from services.robotwin.catalog import source_root


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--tasks',nargs='+');p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--pilot-first',action='store_true',help='Open the remaining GPUs after two successful distinct recordings')
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'.batch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # Presence of the task class avoids treating reward.py as a task.
    import ast
    tasks=[]
    for path in sorted((source_root()/'envs').glob('*.py')):
        if path.stem.startswith('_'):continue
        tree=ast.parse(path.read_text())
        if any(isinstance(node,ast.ClassDef) and node.name==path.stem for node in tree.body):tasks.append(path.stem)
    if args.tasks:
        if set(args.tasks)-set(tasks):raise ValueError('Unknown native task')
        tasks=args.tasks
    pending=[];done=[];failed={};running={};pilot_passed=not args.pilot_first;pilot_task=None
    for task in tasks:
        status=root/task/'aloha-agilex_clean_50/collection_status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='recorded':done.append(task)
        else:pending.append(task)
    manifest={'tasks':tasks,'per_task':args.episodes,'total_target':len(tasks)*args.episodes,'collection':'clean',
              'gpus':args.gpus,'started_unix':time.time(),'status':'running'}
    atomic_json(root/'batch_manifest.json',manifest)
    while pending or running:
        if pilot_task and not pilot_passed:
            status=root/pilot_task/'aloha-agilex_clean_50/collection_status.json'
            if status.exists():
                pilot=json.loads(status.read_text())
                pilot_passed=(pilot.get('completed_recordings',0)>=min(2,args.episodes)
                              and pilot.get('unique_initial_scenes',0)>=min(2,args.episodes))
            if pilot_task in failed:
                manifest.update(status='pilot_failed',failed_tasks=failed)
                atomic_json(root/'batch_manifest.json',manifest)
                raise RuntimeError('Pilot failed; remaining tasks were not launched')
        for gpu in args.gpus:
            if not pilot_passed and (gpu!=args.gpus[0] or pilot_task is not None):continue
            if gpu in running or not pending:continue
            task=pending.pop(0);out=root/task/'aloha-agilex_clean_50';out.mkdir(parents=True,exist_ok=True)
            if args.pilot_first and pilot_task is None:pilot_task=task
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            cmd=[sys.executable,str(Path(__file__).with_name('collect_robotwin_clean50.py')),'--task',task,
                 '--output',str(out),'--episodes',str(args.episodes)]
            log=(out/'collection.log').open('a')
            child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            log.close();running[gpu]=(task,child)
        for gpu,(task,child) in list(running.items()):
            code=child.poll()
            if code is None:continue
            if code==0:done.append(task)
            else:failed[task]=code
            del running[gpu]
        manifest.update(updated_unix=time.time(),pilot_passed=pilot_passed,recorded_tasks=done,failed_tasks=failed,pending_tasks=pending,
                        running={str(g):{'task':t,'pid':p.pid} for g,(t,p) in running.items()})
        atomic_json(root/'batch_manifest.json',manifest)
        time.sleep(5)
    manifest.update(status='recorded' if not failed else 'needs_attention',updated_unix=time.time())
    atomic_json(root/'batch_manifest.json',manifest)


if __name__=='__main__':main()
