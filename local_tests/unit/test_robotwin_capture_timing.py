"""No simulator: exercise finite execution, failure retention and sample identity."""
import json
from types import SimpleNamespace

import pytest

from scripts import diagnose_robotwin_capture_timing as d
from services.robotwin.acceptance_replay import samples_to_trajectory


def pose(x=0.):
    return {a:dict(pose=[x,0.,0.,1.,0.,0.,0.],gripper=1.) for a in ('left','right')}


def clock(trajectory,phase):
    data=d.arrays(trajectory)
    samples=[]
    for i in range(len(data['left']['pose'])):
        s={a:dict(pose=data[a]['pose'][i].tolist(),gripper=float(data[a]['gripper'][i])) for a in ('left','right')}
        samples.append(dict(sample_index=i,command_index=i if phase=='legacy_eef_replay' else None,
                            sample_sha256=d.json_hash(s),physics_step_completed=i*10,simulated_seconds=i*.04))
    return dict(samples=samples,clock_complete=True,timing_compatibility_ready=True)


@pytest.fixture
def harness(tmp_path,monkeypatch):
    trace=samples_to_trajectory([pose(),pose(.01),pose(.02)])
    state={'captures':0,'replays':0,'factories':0,'closed':0,'capture_error':False,
           'clock_complete':True,'native_success':True,'tracking':True}
    ref=dict(trajectory=trace,bundle={},replay_adapter={},provenance=dict(scene_sha256='scene'))
    class Native:
        def close_env(self):state['closed']+=1
    def factory(seed):
        assert seed==10000;state['factories']+=1;return Native()
    class Observer:
        def __init__(self,native,sample,*,phase,**kw):self.phase=phase
        def start(self):return self
        def stop(self,**kw):pass
        def receipt(self):
            result=clock(trace,self.phase)
            result.update(clock_complete=state['clock_complete'],timing_compatibility_ready=state['clock_complete'])
            return result
        def __enter__(self):return self.start()
        def __exit__(self,*args):self.stop()
    def capture(factory,template,seed,excluded,*,attempt_path):
        state['captures']+=1;n=factory(seed)
        try:
            if state['capture_error']:raise ValueError('expert failed')
            return ref
        finally:
            n.close_env();d.atomic_json(attempt_path,dict(status='failed' if state['capture_error'] else 'reference_captured'))
    def replay(native,trajectory):
        state['replays']+=1
        return dict(actual=trace,native_success=state['native_success'],planner_success=True,execution_complete=True)
    monkeypatch.setattr(d,'fresh_physical_state',lambda n:{'state':1})
    monkeypatch.setattr(d,'ControlTimingObserver',Observer)
    monkeypatch.setattr(d.legacy,'capture_reference',capture)
    monkeypatch.setattr(d,'check_initial',lambda *a:{'state':1})
    monkeypatch.setattr(d,'install_base_frame_adapter',lambda n:{})
    monkeypatch.setattr(d,'replay',replay)
    monkeypatch.setattr(d,'tracking_report',lambda *a:dict(realized=state['tracking'],limits=d.TRACKING))
    monkeypatch.setattr(d,'trajectory_violations',lambda *a:{'passed':True})
    def run():return d.run_task(tmp_path,'dump_bin_bigbin',[{'task':'dump_bin_bigbin'}],
                    {'excluded_training_scene_sha256':[]},[],factory)
    return state,run,tmp_path


def test_fixed_one_expert_one_baseline_no_candidates(harness):
    state,run,path=harness;result=run()
    assert result['status']=='diagnostic_complete'
    assert [state[k] for k in ('captures','replays','factories','closed')]==[1,1,2,2]
    assert json.loads((path/'sample_time_comparison.json').read_text())['original_sample_hashes_verified']
    assert not result['acceptance_ready'] and not result['training_allowed']


def test_capture_failure_retained_no_baseline_or_seed_replacement(harness):
    state,run,path=harness;state['capture_error']=True;result=run()
    assert result['status']=='blocked_capture'
    assert [state[k] for k in ('captures','replays','factories','closed')]==[1,0,1,1]
    assert json.loads((path/'expert_attempt.json').read_text())['status']=='failed'
    assert not (path/'legacy_outcome.json').exists()


def test_incomplete_clock_blocks_baseline_without_retry(harness):
    state,run,_=harness;state['clock_complete']=False;result=run()
    assert result['status']=='blocked_capture' and state['replays']==0


@pytest.mark.parametrize('field',['native_success','tracking'])
def test_native_failure_or_strict_tracking_failure_not_retried(harness,field):
    state,run,path=harness;state[field]=False;result=run()
    assert result['status']=='blocked_baseline'
    assert state['replays']==1 and state['factories']==2
    raw=json.loads((path/'legacy_outcome.json').read_text())
    assert raw['native_success']==state['native_success']
    assert raw['realized']==state['tracking'] and raw['tracking']['limits']==d.TRACKING
    assert raw['actual_constraints']['passed'] and not raw['formal_evidence_eligible']


def test_template_routing_is_exact_no_physical_reexecution():
    templates=[{'task':'dump_bin_bigbin','route':0},{'task':'dump_bin_bigbin','route':1}]
    observed=[];selection={}
    def compile(t,*_):
        observed.append(t['route'])
        if t['route']==0:raise d.OutsideTemplate('other signature')
        return {'bundle':True},{'signature':['close']}
    compiled=d.routed_compiler(templates,compile,selection)
    assert compiled(None,{},10000,'scene')[0]=={'bundle':True}
    assert observed==[0,1] and selection['template']['route']==1
    assert len(selection['compilation_rejections'])==1
    with pytest.raises(ValueError,match='exactly one'):
        d.routed_compiler(templates,lambda *_:({}, {'signature':[]}),{})(None,{},10000,'scene')


def test_sample_maps_retain_duplicate_rows_and_detect_forged_indices():
    trace=samples_to_trajectory([pose(),pose(),pose(.1)])
    a,b=clock(trace,'expert_capture'),clock(trace,'legacy_eef_replay')
    a['samples'][1].update(physics_step_completed=0,simulated_seconds=0.)
    result=d.compare_sample_maps(trace,trace,a,b)
    assert result['timing_comparison_valid'] and result['samples'][1]['replay_minus_expert_seconds']==.04
    b['samples'][2]['command_index']=1
    assert not d.compare_sample_maps(trace,trace,a,b)['timing_comparison_valid']


def test_existing_output_rejected_before_native_initialization(tmp_path,monkeypatch):
    plan=tmp_path/'design.json';plan.write_text(json.dumps({'native_commit':'commit','training_root':'train'}))
    output=tmp_path/'existing';output.mkdir();(output/'evidence').write_text('preserve')
    monkeypatch.setattr(d,'validate_design',lambda *_:([],{},[],{}))
    from services.robotwin import runtime
    monkeypatch.setattr(runtime,'source_commit',lambda:'commit')
    monkeypatch.setattr(d.legacy,'training_exclusions',lambda _: {})
    monkeypatch.setattr(runtime,'initialize_runtime',lambda:pytest.fail('must not initialize GPU'))
    with pytest.raises(FileExistsError):
        d.run(SimpleNamespace(design=plan,task='dump_bin_bigbin',output=output,check_only=False,execute=True))
    assert (output/'evidence').read_text()=='preserve'


def test_diagnostic_kind_cannot_be_formal_manifest_even_if_renamed():
    from services.robotwin.acceptance import validate_manifest
    with pytest.raises((KeyError,ValueError,TypeError)):
        validate_manifest(dict(kind=d.KIND,version=2,split='dev',**d.FLAGS))


def synthetic_complete(output):
    """Fake CPU ledger solely for verifier tests; never native evidence."""
    trace=samples_to_trajectory([pose(),pose(.01)])
    d.atomic_json(output/'diagnostic_manifest.json',dict(kind=d.KIND,design_sha256='d'*64,
                    task='dump_bin_bigbin',seed=10000,**d.FLAGS))
    d.atomic_json(output/'run.json',dict(kind=d.KIND,status='diagnostic_complete',task='dump_bin_bigbin',
                    seed=10000,native_episodes_started=2,**d.FLAGS))
    clocks=[]
    for name,phase in [('expert','expert_capture'),('replay','legacy_eef_replay')]:
        receipt=clock(trace,phase)
        events=[]
        events.append(dict(sequence=0,kind='sample',**receipt['samples'][0]))
        events.extend(dict(sequence=i,kind='backend_complete',physics_step_completed=i) for i in range(1,11))
        events.append(dict(sequence=11,kind='sample',**receipt['samples'][1]))
        receipt.update(event_count=len(events),events_sha256=d.json_hash(events),completed_physics_steps=10)
        d.atomic_json(output/(name+'_clock.json'),receipt)
        (output/(name+'_events.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in events))
        clocks.append(receipt)
    d.atomic_json(output/'reference.json',dict(trajectory=trace))
    d.atomic_json(output/'legacy_outcome.json',dict(actual=trace,execution_complete=True,native_success=True,
                    planner_success=True,tracking=d.tracking_report(trace,trace),**d.FLAGS))
    d.atomic_json(output/'sample_time_comparison.json',d.compare_sample_maps(trace,trace,*clocks))
    d.commit_index(output)


def test_complete_output_recomputed_not_just_status(tmp_path):
    synthetic_complete(tmp_path)
    result=d.verify_output(tmp_path,'d'*64)
    assert result['status']=='diagnostic_complete' and not result['acceptance_ready']


@pytest.mark.parametrize('tamper',['orphan','bytes','counter_rehash','native_failure_rehash','sample_index_rehash'])
def test_verify_rejects_tampering_even_if_index_rebuilt(tmp_path,tamper):
    synthetic_complete(tmp_path)
    if tamper=='orphan':(tmp_path/'uncommitted_record.json').write_text('{}')
    elif tamper=='bytes':(tmp_path/'run.json').write_text('{}')
    else:
        name='legacy_outcome.json' if tamper=='native_failure_rehash' else 'replay_clock.json'
        value=d.read(tmp_path/name)
        if tamper=='counter_rehash':value['completed_physics_steps']=11
        elif tamper=='native_failure_rehash':value['native_success']=False
        else:value['samples'][1]['command_index']=0
        d.atomic_json(tmp_path/name,value);d.commit_index(tmp_path)
    with pytest.raises(ValueError):d.verify_output(tmp_path,'d'*64)
