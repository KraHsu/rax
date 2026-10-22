#!/usr/bin/env python3
"""Report recorded and annotated quotas separately, with optional HDF5 audit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def summary(root,audit=False):
    manifest=json.loads((root/'batch_manifest.json').read_text());rows=[]
    for task in manifest['tasks']:
        directory=root/task/'aloha-agilex_clean_50';status=directory/'collection_status.json'
        state=json.loads(status.read_text()) if status.exists() else {'status':'queued'}
        data=list((directory/'data').glob('episode*.hdf5'));raw=list((directory/'raw').glob('episode*.hdf5'))
        errors=[json.loads(p.read_text()) for p in (directory/'annotation_queue').glob('episode*.json')]
        row={'task':task,'status':state['status'],'recorded':len(data)+len(raw),'annotated':len(data),
             'target':manifest['per_task'],'unique_seeds':state.get('unique_seeds',0),
             'unique_initial_scenes':state.get('unique_initial_scenes',0),
             'annotation_errors':sum(r.get('annotation_status')=='error' for r in errors)}
        if 'error' in state:row['error']=state['error']
        if audit and data+raw:
            import h5py
            from scripts.collect_robotwin_clean50 import inventory
            from services.robotwin.constraint_schema import read_embedded
            entries=inventory(directory)
            for path in data:
                with h5py.File(path,'r') as h:
                    bundle=read_embedded(h)
                    if not bundle or not any(c['evidence']['status']=='accepted' for c in bundle['constraints']):
                        raise ValueError(f'Published episode has no accepted constraints: {path}')
            row['audited_episodes']=len(entries)
        rows.append(row)
    return {'target':manifest['total_target'],'recorded':sum(r['recorded'] for r in rows),
            'annotated':sum(r['annotated'] for r in rows),'ready_tasks':sum(r['annotated']==r['target'] for r in rows),
            'batch_status':manifest['status'],'running':manifest.get('running',{}),'tasks':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--audit',action='store_true');args=p.parse_args()
    print(json.dumps(summary(args.root,args.audit),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
