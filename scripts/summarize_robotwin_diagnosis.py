#!/usr/bin/env python3
"""Reduce measured controller/contact evidence and matched continuation trials."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from services.robotwin.annotation import rotation
from services.robotwin.tolerance import measured_calibration_sample


def relative_point(eef_pose, point):
    return rotation(eef_pose[3:]).T @ (np.asarray(point)-eef_pose[:3])


def summarize_trial(record, arm='left'):
    actions=record['action_diagnostics']
    result={k:record[k] for k in ('trial','phase','delta','continuation','outcome','realized')}
    result['tracking_error_m']=record['tracking_error_m']
    result['actions']=[]
    for action in actions:
        state=action['after']; pose=state[arm+'_pose']; request=action['request']
        result['actions'].append({
            'stage':action['stage'],'action':action['action_index'],
            'command':request['action'],'eef_xyz_m':pose[:3],
            'target_xyz_m':request['target_pose'][:3] if request['action']=='move' else None,
            'target_error_m':(np.array(pose[:3])-request['target_pose'][:3]).tolist() if request['action']=='move' else None,
            'joint_tracking_error_by_name':{name:j['qpos']-j['target'] for name,j in state['joints'].items()
                                            if abs(j['qpos']-j['target'])>1e-4},
            'command_opening':state[arm+'_opening'],
            'physical_gripper_fraction':state['physical_gripper_fraction'],
            'bottle_functional_xyz_m':state['bottle_functional_pose'][:3],
            'bottle_in_eef_m':relative_point(pose,state['bottle_functional_pose'][:3]).tolist(),
            'contacts':[{k:c[k] for k in ('bodies','impulse_Ns')} for c in state['contacts']]})
    if actions:
        last=actions[-1]['after']; held=record['hold_state']
        result['terminal_bottle_height_m']=last['bottle_functional_pose'][2]
        result['held_bottle_height_m']=held['bottle_functional_pose'][2]
        lifted=[a['after'] for a in actions if a['stage']==1]
        if lifted:
            local=relative_point(lifted[-1][arm+'_pose'],lifted[-1]['bottle_functional_pose'][:3])
            predicted=rotation(last[arm+'_pose'][3:])@local+last[arm+'_pose'][:3]
            result['rigid_grasp_predicted_terminal_xyz_m']=predicted.tolist()
            result['rigid_grasp_prediction_error_m']=float(np.linalg.norm(predicted-last['bottle_functional_pose'][:3]))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    records=[json.loads(line) for line in (args.run/'results.jsonl').read_text().splitlines()]
    arm=json.loads((args.run/'expert_program.json').read_text())[0][0]['arm']
    summaries=[summarize_trial(r,arm) for r in records]
    pairs={}
    for r in summaries:
        if not r['phase'].startswith('paired'):continue
        key=json.dumps(r['delta'],sort_keys=True)
        bucket=pairs.setdefault(key,{'delta':r['delta']})
        bucket.setdefault(r['continuation'],[]).append({'trial':r['trial'],'outcome':r['outcome'],
            'height_m':r.get('terminal_bottle_height_m'),'held_height_m':r.get('held_bottle_height_m')})
    reference=json.loads((args.run/'reference.json').read_text())
    samples=[measured_calibration_sample(r,reference) for r in records]
    payload={'run':str(args.run),'pairs':list(pairs.values()),'trials':summaries,
             'measured_feasible_samples':[s for s in samples if s is not None]}
    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False))
    print(json.dumps({'trials':len(summaries),'output':str(args.output)}))


if __name__=='__main__':main()
