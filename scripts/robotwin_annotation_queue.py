#!/usr/bin/env python3
"""H200-side claim/review/publish transactions for local RAX annotation workers."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.robotwin.collection import atomic_json
from services.robotwin.annotation import export_review,embed_review


def claim(root,worker):
    root=Path(root).resolve();selected=None
    control=root/'annotation_control.json'
    control=json.loads(control.read_text()) if control.exists() else {}
    if control.get('pause_agent_claims') or worker in control.get('paused_workers',[]):
        return {'waiting':True,'complete':False,'paused':True}
    with (root/'.annotation_queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        files=sorted(root.glob('*/aloha-agilex_clean_50/annotation_queue/episode*.json'),
                     key=lambda p:(int(p.stem.removeprefix('episode')),str(p)))
        busy_tasks=set()
        for path in files:
            receipt=json.loads(path.read_text())
            if (receipt['annotation_status']=='processing' and receipt.get('claimed_by')!='programmatic'
                    and time.time()-receipt.get('claimed_unix',0)<3600):busy_tasks.add(receipt['task'])
        for path in files:
            receipt=json.loads(path.read_text());state=receipt['annotation_status']
            if state=='done' or receipt.get('annotation_attempts',0)>=3:continue
            if state=='processing' and time.time()-receipt.get('claimed_unix',0)<3600:continue
            if not Path(receipt['path']).exists():continue
            if control.get('task_templates'):
                if receipt['task'] in busy_tasks:continue
                from services.robotwin.annotation_templates import match
                if not receipt.get('programmatic_error'):
                    try:
                        if match(root,receipt['task'],receipt['path'],include_candidates=True):continue
                    except (ValueError,KeyError):pass  # RAX must inspect an invalid/uncovered case.
            job=uuid.uuid4().hex
            receipt.update(annotation_status='processing',claimed_by=worker,claim_id=job,claimed_unix=time.time(),
                           annotation_attempts=receipt.get('annotation_attempts',0)+1)
            atomic_json(path,receipt)
            review=root/'reviews'/receipt['task']/f'episode{receipt["index"]}-{job}'
            selected={'receipt':str(path),'episode':receipt['path'],'claim_id':job,'review':str(review),
                      'task':receipt['task'],'index':receipt['index']};break
    if selected:
        export_review(selected['episode'],selected['review'])
        return selected
    manifest=json.loads((root/'batch_manifest.json').read_text())
    done=len(list(root.glob('*/aloha-agilex_clean_50/data/episode*.hdf5')))
    return {'waiting':True,'published':done,'target':manifest['total_target'],
            'complete':done==manifest['total_target']}


def complete(receipt_path,claim_id,proposal,agent_run,template=None):
    receipt_path=Path(receipt_path);receipt=json.loads(receipt_path.read_text())
    if receipt.get('claim_id')!=claim_id or receipt['annotation_status']!='processing':raise ValueError('Stale annotation lease')
    if not any(c.get('evidence',{}).get('status')=='accepted' for c in proposal.get('constraints',[])):
        raise ValueError('No accepted constraints; raw recording remains outside data/')
    episode=Path(receipt['path']);target=episode.parent.parent/'data'/episode.name
    if target.exists():raise FileExistsError(target)
    result=embed_review(episode,proposal,agent_run=agent_run)
    if result['accepted']<1:raise ValueError('Annotation did not pass reference validation')
    target.parent.mkdir(exist_ok=True);episode.rename(target)
    receipt.update(annotation_status='done',path=str(target),annotation_result=result,finished_unix=time.time())
    receipt['annotation_result']['episode']=str(target)
    atomic_json(receipt_path,receipt)
    response={'published':str(target),'accepted':result['accepted'],'revision':result['revision']}
    if template is not None:
        from services.robotwin.annotation_templates import install
        try:
            registered=install(receipt_path.parents[3],template,source_episode=target,agent_run=agent_run)
            response['template_status']=registered['status']
        except Exception as exc:
            # A valid published annotation remains valid if its reusable rule fails.
            receipt['template_error']=f'{type(exc).__name__}: {exc}'
            atomic_json(receipt_path,receipt);response['template_error']=receipt['template_error']
    return response


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('claim');a.add_argument('--root',type=Path,required=True);a.add_argument('--worker',required=True)
    a=sub.add_parser('complete');a.add_argument('--receipt',required=True);a.add_argument('--claim-id',required=True)
    a.add_argument('--proposal',type=Path,required=True);a.add_argument('--agent-run',required=True)
    a.add_argument('--template',type=Path)
    a=sub.add_parser('fail');a.add_argument('--receipt',required=True);a.add_argument('--claim-id',required=True);a.add_argument('--error',required=True)
    args=p.parse_args()
    if args.command=='claim':result=claim(args.root,args.worker)
    elif args.command=='complete':result=complete(args.receipt,args.claim_id,json.loads(args.proposal.read_text()),args.agent_run,
                                                json.loads(args.template.read_text()) if args.template else None)
    else:
        path=Path(args.receipt);receipt=json.loads(path.read_text())
        if receipt.get('claim_id')!=args.claim_id or receipt['annotation_status']!='processing':raise ValueError('Stale lease')
        receipt.update(annotation_status='error',annotation_error=args.error);atomic_json(path,receipt);result={'failed':True}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
