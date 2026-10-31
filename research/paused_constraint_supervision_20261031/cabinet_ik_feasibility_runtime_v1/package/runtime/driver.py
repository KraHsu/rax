"""Finite standalone IK driver. Importing this file never imports cuRobo/Torch/SAPIEN."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import time
from contracts import VERSION,FLAGS,SOLVER_FIELDS,inputs,read,digest,atomic,analyze


def preflight(package,external=True):
    design,targets,limits=inputs(package,external)
    from source_audit import check
    source_audit=check(package)
    return dict(version=VERSION,slots=12,seeds_per_call=32,native_initializations=0,scenes=0,
                expert_attempts=0,replays=0,gpu_started=False,limits=limits.tolist(),source_audit=source_audit,**FLAGS)


def execute(package,output,backend_factory,clock=time.monotonic):
    package=Path(package);output=Path(output)
    design,targets,limits=inputs(package,True)
    output.mkdir(parents=True)  # one shot; no resume and no pre-existing output
    started=clock();rows=[];counts=dict(solve_attempts=0,constraint_evaluations=0,fk_evaluations=0)
    state=dict(version=VERSION,status='initializing',plan_sha256=digest(package/'plan.json'),design_sha256=digest(package/'design/probe_design.json'),
               targets_sha256=digest(package/'design/targets.json'),counts=counts,records=[],
               scenes=0,native_initializations=0,expert_attempts=0,replays=0,automatic_retries=0,**FLAGS)
    atomic(output/'intent.json',dict(state,deadline_seconds=900))
    atomic(output/'state.json',state)
    backend=None
    try:
        # Actual imports/initialization begin only inside the authorized child execution path.
        backend=backend_factory(package,output,limits)
        for slot in design['slots']:
            if clock()-started>=900:raise TimeoutError('Execution deadline; no retry')
            mode=next(x for x in design['modes'] if x['id']==slot['mode']);target=targets[slot['target']]
            folder=output/'slots'/slot['slot_id'];folder.mkdir(parents=True)
            atomic(folder/'intent.json',dict(slot,target=target,mode_definition=mode,counts_before=dict(counts)))
            backend.prepare(mode)
            if counts['solve_attempts']>=12:raise RuntimeError('Solve budget exhausted')
            counts['solve_attempts']+=1;state['status']='solving';atomic(output/'state.json',state)
            before=clock()
            def charged(kind):
                if kind not in ('constraint_evaluations','fk_evaluations') or counts[kind]>=counts['solve_attempts']:raise RuntimeError('Post-evaluation budget exceeded')
                counts[kind]+=1;atomic(output/'state.json',state)
            raw=backend.solve(target,charged)
            atomic(folder/'raw.json',raw)
            report=analyze(package,target,raw)
            evidence={name:digest(folder/name) for name in ('intent.json','raw.json','seed_matrix.json','solver_result.json','post_constraints.json')}
            record=dict(slot,elapsed_seconds=clock()-before,raw_sha256=digest(folder/'raw.json'),evidence_sha256=evidence,report=report)
            atomic(folder/'record.json',record);rows.append(record)
            state['records']=[dict(slot_id=r['slot_id'],record_sha256=digest(output/'slots'/r['slot_id']/'record.json')) for r in rows]
            global_files=['runtime_identity.json','solver_entries.json']+['modes/'+m+'.json' for m in dict.fromkeys(r['mode'] for r in rows)]
            state['runtime_evidence_sha256']={name:digest(output/name) for name in global_files}
            atomic(output/'state.json',state)
            if slot['slot_id']=='ik_00' and not report['any_reported_success']:
                state['status']='blocked_known_fk_control';atomic(output/'state.json',state);return state
        state['status']='completed_diagnostic';atomic(output/'state.json',state)
        return state
    except BaseException as exc:
        state['status']='incomplete';state['error']=dict(type=type(exc).__name__,message=str(exc))
        atomic(output/'state.json',state)
        raise
    finally:
        primary_error=sys.exc_info()[1]
        if backend is not None:
            try:backend.close()
            except BaseException as exc:
                state['cleanup_error']=dict(type=type(exc).__name__,message=str(exc));state['status']='incomplete';atomic(output/'state.json',state)
                if primary_error is None:raise


def verify_output(output,_design_hash=None,design=None):
    output=Path(output);package=Path(__file__).resolve().parents[1]
    frozen,targets,_=inputs(package,True);state=read(output/'state.json');intent=read(output/'intent.json')
    if state.get('plan_sha256')!=digest(package/'plan.json') or state['version']!=VERSION or state['design_sha256']!=digest(package/'design/probe_design.json') or state['targets_sha256']!=digest(package/'design/targets.json'):raise ValueError('Run input/version mismatch')
    for key in ('scenes','native_initializations','expert_attempts','replays','automatic_retries'):
        if state[key]!=0 or intent[key]!=0:raise ValueError('Forbidden native budget')
    if any(state[k]is not v for k,v in FLAGS.items()):raise ValueError('Invalid acceptance label')
    if 'error'in state or 'cleanup_error'in state:raise ValueError('Failed/cleanup-error evidence cannot be completed')
    records=state['records'];n=len(records)
    mode_ids=list(dict.fromkeys(x['mode'] for x in frozen['slots'][:n]))
    expected_global={'runtime_identity.json','solver_entries.json'}|{'modes/'+m+'.json' for m in mode_ids}
    if set(state.get('runtime_evidence_sha256',{}))!=expected_global or any(digest(output/name)!=h for name,h in state['runtime_evidence_sha256'].items()):raise ValueError('Runtime evidence SHA closure incomplete')
    if read(output/'solver_entries.json')!=dict(solve_single_started=n,unexpected_entries=0) or (output/'unexpected_solver_entry.json').exists():raise ValueError('Actual solver entry count differs')
    identity=read(output/'runtime_identity.json')
    if identity.get('device')!='cuda:0' or identity.get('visible_devices')!='0' or identity.get('scenes')!=0 or identity.get('native_task_imports')is not False:raise ValueError('Runtime identity differs')
    for mode_id in mode_ids:
        if read(output/'modes'/f'{mode_id}.json')['mode']!=next(x for x in frozen['modes'] if x['id']==mode_id):raise ValueError('Actual collision mode differs')
    expected={x['slot_id'] for x in frozen['slots'][:n]}
    present={p.parent.name for p in (output/'slots').glob('*/record.json')}
    if present!=expected:raise ValueError('Committed record set differs')
    if {p.name for p in (output/'slots').iterdir()}!=expected:raise ValueError('Slot directory set differs from committed prefix')
    for slot,row in zip(frozen['slots'],records,strict=False):
        folder=output/'slots'/slot['slot_id'];record=read(folder/'record.json')
        if row['slot_id']!=slot['slot_id'] or digest(folder/'record.json')!=row['record_sha256'] or record['raw_sha256']!=digest(folder/'raw.json'):raise ValueError('Record SHA/prefix mismatch')
        if set(record['evidence_sha256'])!={'intent.json','raw.json','seed_matrix.json','solver_result.json','post_constraints.json'}:raise ValueError('Missing raw proof component')
        if any(digest(folder/name)!=h for name,h in record['evidence_sha256'].items()):raise ValueError('Raw proof SHA mismatch')
        target=targets[slot['target']];mode=next(x for x in frozen['modes'] if x['id']==slot['mode'])
        if read(folder/'intent.json')!=dict(slot,target=target,mode_definition=mode,counts_before=dict(solve_attempts=int(slot['slot_id'][3:]),constraint_evaluations=int(slot['slot_id'][3:]),fk_evaluations=int(slot['slot_id'][3:]))):raise ValueError('Original slot intent mismatch')
        raw=read(folder/'raw.json');partial=read(folder/'solver_result.json')
        if set(partial)!=SOLVER_FIELDS or read(folder/'seed_matrix.json')!=raw['generated_seeds'] or any(raw.get(k)!=v for k,v in partial.items()) or read(folder/'post_constraints.json')!={k:v for k,v in raw.items() if k.startswith('constraint')}:raise ValueError('Partial/raw evidence inconsistent')
        if any(record[k]!=v for k,v in slot.items()) or record['report']!=analyze(package,target,raw):raise ValueError('Derived report mismatch')
    if state['status']=='completed_diagnostic':
        if n!=12 or state['counts']!=dict(solve_attempts=12,constraint_evaluations=12,fk_evaluations=12):raise ValueError('Incomplete twelve-slot budget')
    elif state['status']=='blocked_known_fk_control':
        if n!=1 or state['counts']!=dict(solve_attempts=1,constraint_evaluations=1,fk_evaluations=1) or read(output/'slots/ik_00/record.json')['report']['any_reported_success']:raise ValueError('Invalid control block')
    else:raise ValueError('Incomplete evidence; never retry')
    return dict(state,raw_evidence_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*.json'))})


def run(args):
    package=Path(args.design).resolve().parents[1]
    result=preflight(package,external=True)
    if args.check_only:
        print(__import__('json').dumps(result,sort_keys=True));return 0
    if not args.execute:raise ValueError('Explicit execute required')
    from gpu_backend import Backend
    state=execute(package,args.output,Backend)
    return 0 if state['status'] in ('completed_diagnostic','blocked_known_fk_control') else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--design',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--check-only',action='store_true');p.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.check_only and args.execute:p.error('Choose check-only or execute')
    raise SystemExit(run(args))
