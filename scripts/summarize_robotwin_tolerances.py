#!/usr/bin/env python3
"""Summarize tolerance evidence without promoting sampled points to a feasible set."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.robotwin.tolerance import bracket


def joint_profile(root):
    profile={'role':'calibration_evidence','task':'adjust_bottle',
             'claim':'Sampled joint tests only; no continuous-set or cross-task guarantee.',
             'space':'Requested offsets to frozen expert targets, audited against measured geometry.',
             'training_label_update':'Evidence only; command offsets are not automatic measured-trajectory loss bounds.',
             'tracking_criterion':'Changed grasp coordinates max(1 mm,10% of offset), unchanged coordinates 2 mm; explicit zero coordinates in joint cases use 1 mm. Terminal Z max(1 mm,10%).',
             'scenes':[]}
    all_joint=[]
    for q in sorted(root.glob('tolerance_seed*_confirm')):
        metadata=json.loads((q/'run.json').read_text())
        if metadata['status']!='completed':continue
        p=root/q.name.replace('_confirm','_v2')
        original=[json.loads(x) for x in (p/'results.jsonl').read_text().splitlines()]
        confirmed=[json.loads(x) for x in (q/'results.jsonl').read_text().splitlines()]
        if json.loads((p/'expert_program.json').read_text())!=json.loads((q/'expert_program.json').read_text()):
            raise ValueError('Confirmation program differs from scan')
        joint=[r for r in confirmed if r['phase']=='confirmation' and len(r['delta'])>1]
        if not joint:continue
        all_joint.extend(joint);reference=json.loads((p/'reference.json').read_text())
        event=np.array([r['features']['closure_xyz_m'] for r in joint])-reference['closure_xyz_m']
        ends=np.array([r['features']['terminal_xyz_m'][2] for r in joint])-reference['terminal_xyz_m'][2]
        profile['scenes'].append({'seed':metadata['seed'],'reference':reference,
            'program_sha256':hashlib.sha256((p/'expert_program.json').read_bytes()).hexdigest(),
            'joint_cases':len(joint),'joint_passes':sum(r['outcome']=='success' and r['realized'] for r in joint),
            'measured_closure_event_delta_min_m':event.min(0).tolist(),
            'measured_closure_event_delta_max_m':event.max(0).tolist(),
            'measured_terminal_z_delta_m':[float(ends.min()),float(ends.max())],
            'max_tracking_error_m':{k:max(abs(r['tracking_error_m'].get(k,0)) for r in joint)
                for k in ('grasp_x','grasp_y','grasp_z','terminal_z')},
            'combined_brackets':{a:{str(s):bracket(original+confirmed,a,s)
                    for s in ((1,) if a=='opening' else (-1,1))}
                    for a in ('grasp_x','grasp_y','grasp_z','opening','terminal_z')}})
    profile['requested_joint_domain']={k:[min(r['delta'][k] for r in all_joint),max(r['delta'][k] for r in all_joint)]
                                      for k in all_joint[0]['delta']} if all_joint else {}
    profile['units']='World metres for grasp_x/y/z and terminal_z; unitless command fraction for opening.'
    return profile


def summarize(root):
    studies=[];rows=[]
    for path in sorted(root.glob('tolerance_seed*')):
        if not path.is_dir() or not (path/'results.jsonl').exists():continue
        metadata=json.loads((path/'run.json').read_text())
        records=[json.loads(line) for line in (path/'results.jsonl').read_text().splitlines()]
        summary={'run':path.name,'metadata':metadata,'trials':len(records),
                 'source_results_sha256':hashlib.sha256((path/'results.jsonl').read_bytes()).hexdigest(),
                 'native_task_successes':sum(r.get('native_success',False) for r in records),
                 'successful_trials':sum(r['outcome']=='success' for r in records),
                 'planner_failures':sum(r['outcome']=='planner_failure' for r in records),
                 'unrealized_trials':sum(r.get('realized') is False for r in records),
                 'baseline_outcomes':[r['outcome'] for r in records if r['phase'].startswith('baseline')]}
        if path.name.endswith('_v2'):
            summary['replicated_command_brackets']={a:{str(s):bracket(records,a,s)
                    for s in ((1,) if a=='opening' else (-1,1))}
                    for a in ('grasp_x','grasp_y','grasp_z','opening','terminal_z')}
        if path.name.endswith('_confirm'):
            joint=[r for r in records if r['phase']=='confirmation' and len(r['delta'])>1]
            summary['joint_confirmation']={'trials':len(joint),
                'successful':sum(r['outcome']=='success' for r in joint),
                'successful_and_realized':sum(r['outcome']=='success' and r['realized'] for r in joint),
                'failed_trial_ids':[r['trial'] for r in joint if r['outcome']!='success'],
                'unrealized_trial_ids':[r['trial'] for r in joint if not r['realized']]}
        studies.append(summary)
        for r in records:
            row={'run':path.name,'seed':metadata['seed'],'trial':r['trial'],'phase':r['phase'],
                 'delta_json':json.dumps(r['delta'],sort_keys=True),'outcome':r['outcome'],
                 'native_success':r.get('native_success'),'hold_success':r.get('hold_success'),
                 'realized':r.get('realized'),
                 'measured_offset_json':json.dumps(r.get('measured_offset_m',{}),sort_keys=True)}
            rows.append(row)
    return {'scope':'Sampled empirical evidence conditional on frozen expert targets, clean scenes and native replanning; not a continuous-set guarantee.',
            'studies':studies,'logged_trials':sum(s['trials'] for s in studies)},rows


def plot_tracking(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    paths=sorted(root.glob('tolerance_seed*_v2'))
    fig,axs=plt.subplots(1,len(paths),figsize=(12,4),squeeze=False,sharex=True,sharey=True)
    extent=[]
    for path in paths:
        for line in (path/'results.jsonl').read_text().splitlines():
            r=json.loads(line)
            if set(r['delta'])=={'grasp_z'}:extent.append(r['delta']['grasp_z']*1000)
    limits=(min(extent)-5,max(extent)+5)
    for ax,path in zip(axs[0],paths):
        records=[json.loads(x) for x in (path/'results.jsonl').read_text().splitlines()]
        for r in records:
            if set(r['delta'])!={'grasp_z'} or 'grasp_z' not in r.get('measured_offset_m',{}):continue
            color='#8c8c8c' if not r['realized'] else '#218739' if r['outcome']=='success' else '#c0392b'
            ax.scatter(r['delta']['grasp_z']*1000,r['measured_offset_m']['grasp_z']*1000,c=color,s=30,alpha=.75)
        ax.plot(limits,limits,'--',color='#555555',linewidth=1)
        ax.set(title=f"Clean seed {json.loads((path/'run.json').read_text())['seed']}",xlabel='Requested grasp Z offset (mm)',xlim=limits,ylim=limits)
        ax.grid(alpha=.2)
    axs[0,0].set_ylabel('Measured offset after closure (mm)')
    fig.suptitle('Command offsets and achieved geometry differ under contact')
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='',color=c,label=l) for c,l in
        [('#218739','Success + tracking passed'),('#c0392b','Task failure + tracking passed'),('#8c8c8c','Vector tracking check failed')]],loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.1,1,.93));fig.savefig(output/'grasp_z_tracking.png',dpi=180);fig.savefig(output/'grasp_z_tracking.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result,rows=summarize(args.root)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    (args.output/'joint_candidate_profile.json').write_text(json.dumps(joint_profile(args.root),indent=2,ensure_ascii=False))
    if rows:
        with (args.output/'trials.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if args.plot:plot_tracking(args.root,args.output)
    print(json.dumps({'logged_trials':result['logged_trials'],'runs':len(result['studies']),'output':str(args.output)}))


if __name__=='__main__':main()
