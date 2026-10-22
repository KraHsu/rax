#!/usr/bin/env python3
"""Local persistent worker: small remote review packages -> RAX -> checked publish."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from services.robotwin.collection import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--worker',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--host',default='root@vr.turbo-ai.com');p.add_argument('--port',type=int,default=20379)
    p.add_argument('--remote-project',default='/mnt/data/liangjiaqi/rax_train/rax')
    p.add_argument('--remote-python',default='/mnt/data/liangjiaqi/yzm/miniconda3/envs/RoboTwin/bin/python')
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(root/'.worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    socket=Path.home()/'.cache/rax-validation'/f'annotation-{args.worker}.sock'
    ssh=['ssh','-p',str(args.port),'-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
         '-o','ControlMaster=auto','-o','ControlPersist=600','-o',f'ControlPath={socket}']
    def remote(command):
        completed=subprocess.run([*ssh,args.host,shlex.join(command)],text=True,capture_output=True,timeout=300,check=True)
        return json.loads(completed.stdout)
    command=[args.remote_python,args.remote_project+'/scripts/robotwin_annotation_queue.py']
    previous=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    status={'worker':args.worker,'completed':previous.get('completed',0),'failed':previous.get('failed',0),'status':'starting'}
    while True:
        if (root/'stop_after_current').exists():
            status.update(status='stopped',updated_unix=time.time());atomic_json(root/'status.json',status);return
        job=None
        try:
            job=remote([*command,'claim','--root',args.dataset,'--worker',args.worker])
            if job.get('waiting'):
                status.update(status='complete' if job['complete'] else 'waiting',queue=job,updated_unix=time.time())
                atomic_json(root/'status.json',status)
                if job['complete']:return
                time.sleep(30);continue
            local=root/job['task']/f'episode{job["index"]}-{job["claim_id"]}';local.mkdir(parents=True)
            review=local/'review'
            subprocess.run(['rsync','-az','-e',shlex.join(ssh),args.host+':'+job['review']+'/',str(review)+'/'],check=True,timeout=300)
            status.pop('queue',None)
            status.update(status='annotating',job=job,updated_unix=time.time());atomic_json(root/'status.json',status)
            with (local/'agent.log').open('w') as log:
                subprocess.run([sys.executable,str(PROJECT/'scripts/run_robotwin_annotation.py'),'--review',str(review),
                                '--output',str(local/'agent'),'--timeout','1800','--task-template'],stdout=log,stderr=subprocess.STDOUT,check=True)
            proposal=local/'agent/agent-workspace/annotations.json';remote_proposal=job['review']+'/annotations.json'
            subprocess.run(['rsync','-az','-e',shlex.join(ssh),str(proposal),args.host+':'+remote_proposal],check=True,timeout=300)
            template=local/'agent/agent-workspace/task_template.json'
            report=local/'agent/agent-workspace/REPORT.md'
            subprocess.run(['rsync','-az','-e',shlex.join(ssh),str(template),str(report),args.host+':'+job['review']+'/'],check=True,timeout=300)
            result=remote([*command,'complete','--receipt',job['receipt'],'--claim-id',job['claim_id'],
                           '--proposal',remote_proposal,'--agent-run',job['review'],
                           '--template',job['review']+'/task_template.json'])
            atomic_json(local/'publication.json',result);status['completed']+=1
            print(json.dumps({'event':'published','job':job,**result}),flush=True)
        except Exception as exc:
            status['failed']+=1;status.update(status='retrying',error=f'{type(exc).__name__}: {exc}',updated_unix=time.time())
            print(json.dumps(status),flush=True)
            if job and 'receipt' in job:
                try:remote([*command,'fail','--receipt',job['receipt'],'--claim-id',job['claim_id'],'--error',str(exc)[:1000]])
                except Exception:pass
            time.sleep(30)
        finally:atomic_json(root/'status.json',status)


if __name__=='__main__':main()
