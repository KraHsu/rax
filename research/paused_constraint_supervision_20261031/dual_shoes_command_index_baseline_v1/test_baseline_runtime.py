import copy
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('shoes_baseline',ROOT/'package/runtime/baseline_driver.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
DESIGN=m.read(m.PACKAGE/'design.json')
REF=m.read(m.contained(DESIGN['reference']))
TERMINAL=m.read(m.PACKAGE/'source_evidence/expert_terminal_evidence.json')
PHYSICAL=m.read(m.PACKAGE/'source_evidence/expert_final_physical_state.json')


def observation(success=True):
    roles=copy.deepcopy(TERMINAL['named_roles']);physical=copy.deepcopy(PHYSICAL)
    if not success:
        roles['left_shoe']['p'][0]=.2
        physical['actors'][roles['left_shoe']['physical_actor_index']]['p'][0]=.2
    return dict(boundary='after_existing_replay_final_check_before_cleanup',physical=physical,
        physical_after_sha256=m.json_digest(physical),named_roles=roles,
        gripper_operands=TERMINAL['native_gripper_operands'],
        component_result=m.terminal.predicate_components(roles,TERMINAL['native_gripper_operands'],DESIGN['terminal_policy']['numpy_version']),
        predicate_calls_added=0,physics_steps_added=0,controls_added=0)


def raw_replay(success=True):
    return dict(actual=copy.deepcopy(REF['trajectory']),native_success=success,ever_success=success,final_success=success,
        execution_complete=True,planner_success=True,planner_results=[dict(frame=i,arm=arm,status='Success')for i in range(1,246)for arm in ('left','right')],
        alignment=m.ALIGNMENT,native_success_rule=m.SUCCESS_RULE,error=None,early_termination_suppressed_for_complete_alignment=True)


@pytest.fixture
def simulated(monkeypatch,tmp_path):
    # CPU protocol fixture: no simulator, solver, or GPU; physical evidence is a fixed copied success fixture.
    instances=[];outcomes=[True,True]
    class Native:
        def __init__(self):self.index=len(instances);instances.append(self);self.closed=False
        def close_env(self):self.closed=True
    def factory(seed):assert seed==10000;return Native()
    monkeypatch.setattr(m,'check_initial',lambda n,r:dict(scene=REF['initial_scene'],first_eef={arm:{'pose':REF['trajectory'][arm]['pose'][0],'gripper':REF['trajectory'][arm]['gripper'][0]}for arm in ('left','right')},physical={'synthetic_cpu_fixture':True},source_full_physical_comparison_available=False,exact_available_scene_match=True))
    monkeypatch.setattr(m,'install_base_frame_adapter',lambda n:REF['replay_adapter'])
    monkeypatch.setattr(m,'replay',lambda n,c:raw_replay(outcomes[n.index]))
    monkeypatch.setattr(m,'terminal_observation',lambda n,raw,policy:observation(outcomes[n.index]))
    output=tmp_path/'run';output.mkdir();m.atomic_json(output/'manifest.json',m.expected_manifest(DESIGN))
    return output,factory,instances,outcomes


def refresh(output):
    for p in (output/'trials').glob('*/record.json'):
        record=m.read(p);record['files']={n:m.digest(p.parent/n)for n in record['files']};m.atomic_json(p,record)
    m.atomic_json(output/'artifact_index.json',dict(kind=m.KIND,files={str(p.relative_to(output)):m.digest(p)for p in output.rglob('*')if p.is_file()and p.name!='artifact_index.json'}))


def test_original_246_source_83_rules_recompile_and_self_pass():
    ref,audit=m.validate_design(DESIGN,external=False,numeric=False)
    assert ref==REF and audit['frames']==246 and len(audit['self_constraint_score']['constraints'])==83
    assert audit['self_constraint_score']['max_violation']==0
    assert audit['expert_full_initial_state_present']is False and audit['physics_timestamps_present']is False


def test_two_conditional_baselines_do_not_call_expert(simulated):
    output,factory,natives,_=simulated
    result=m.execute_sequence(output,REF,DESIGN,factory)
    assert result['baseline_reliable']and result['baseline_replays']==2 and result['new_experts']==0
    assert len(natives)==2 and all(n.closed for n in natives)
    assert all(result[k]is False for k in m.FLAGS)


def test_first_native_failure_is_baseline_not_variant_and_stops(simulated):
    output,factory,natives,outcomes=simulated;outcomes[0]=False
    result=m.execute_sequence(output,REF,DESIGN,factory)
    assert result['status']=='blocked_baseline_qualification'and len(natives)==1
    row=result['rows'][0]
    assert row['eligible']and row['commanded_constraint_pass']and row['actual_constraint_pass']and not row['native_success']
    assert result['variants']==0


def test_setup_exception_commits_one_intent_and_never_retries(simulated):
    output,_,_,_=simulated;calls=[]
    def factory(seed):calls.append(seed);raise RuntimeError('setup unavailable')
    result=m.execute_sequence(output,REF,DESIGN,factory)
    assert calls==[10000]and result['baseline_attempts']==1 and result['baseline_replays']==0 and not result['rows'][0]['execution_complete']
    assert (output/'trials/baseline_0/outcome.json').exists()


def test_cleanup_failure_preserved_and_stops(simulated):
    output,factory,natives,_=simulated
    def bad_factory(seed):
        n=factory(seed)
        def fail():raise RuntimeError('cleanup failed')
        n.close_env=fail;return n
    result=m.execute_sequence(output,REF,DESIGN,bad_factory)
    assert len(natives)==1 and result['rows'][0]['errors'][0]['phase']=='cleanup'
    assert result['rows'][0]['native_success']is True and not result['baseline_reliable']


def test_observer_failure_does_not_erase_raw_or_skip_cleanup(simulated,monkeypatch):
    output,factory,natives,_=simulated
    def bad(*a):raise ValueError('observer unavailable')
    monkeypatch.setattr(m,'terminal_observation',bad)
    result=m.execute_sequence(output,REF,DESIGN,factory)
    assert len(natives)==1 and natives[0].closed and result['rows'][0]['execution_complete']
    assert not result['baseline_reliable']and result['rows'][0]['native_success']


@pytest.mark.parametrize('key,value',[('new_experts',1),('baseline_replays_max',3),('variants',1),('physics_time_preserved',True),('automatic_retries',True),('task_coverage_qualified',True)])
def test_budget_identity_rejects(key,value):
    design=copy.deepcopy(DESIGN);design[key]=value
    with pytest.raises(ValueError,match='Fixed contract changed'):m.validate_design(design,external=False,numeric=False)


@pytest.mark.parametrize('change',['hold','predicate','adapter','timing','extra','planner'])
def test_tamper_with_rebuilt_index_rejected(simulated,change):
    output,factory,_,_=simulated;m.execute_sequence(output,REF,DESIGN,factory)
    directory=output/'trials/baseline_0'
    if change=='hold':
        p=directory/'trajectory.json';d=m.read(p);d['commanded']['left']['pose'].append(d['commanded']['left']['pose'][-1]);m.atomic_json(p,d)
    elif change in ('predicate','adapter','planner'):
        p=directory/'outcome.json';d=m.read(p)
        if change=='predicate':d['ever_success']=False
        elif change=='adapter':d['replay_adapter']['arms']['left']['xyz'][0]+=.01
        else:d['planner_results'][0]['status']='Fail'
        m.atomic_json(p,d)
    elif change=='timing':
        p=output/'manifest.json';d=m.read(p);d['physics_time_preserved']=True;m.atomic_json(p,d)
    else:(directory/'unindexed.json').write_text('{}')
    refresh(output)
    with pytest.raises(ValueError):m.verify_output(output,DESIGN)


def test_no_resume_existing_trial(simulated):
    output,factory,_,_=simulated;m.execute_sequence(output,REF,DESIGN,factory)
    with pytest.raises(FileExistsError):m.execute_sequence(output,REF,DESIGN,factory)


def test_tracking_invalid_success_stops_before_second(simulated,monkeypatch):
    output,factory,natives,_=simulated
    def wrong_tracking(n,c):
        raw=raw_replay(True);raw['actual']['left']['pose'][5][0]+=.006;return raw
    monkeypatch.setattr(m,'replay',wrong_tracking)
    report=m.execute_sequence(output,REF,DESIGN,factory)
    assert len(natives)==1 and report['rows'][0]['native_success']and not report['rows'][0]['realized']
    assert report['baseline_replays']==1 and not report['baseline_reliable']


def test_both_baselines_fixed_prefix_required_even_after_rehash(simulated):
    output,factory,natives,outcomes=simulated
    m.execute_sequence(output,REF,DESIGN,factory)
    directory=output/'trials/baseline_0';raw=m.read(directory/'outcome.json');raw.update(native_success=False,ever_success=False,final_success=False)
    obs=m.read(directory/'observations.json');obs['terminal']=observation(False)
    m.atomic_json(directory/'outcome.json',raw);m.atomic_json(directory/'observations.json',obs)
    record=m.read(directory/'record.json');record['derived']=m.derive_trial(raw,m.read(directory/'trajectory.json'),obs,REF,DESIGN);m.atomic_json(directory/'record.json',record)
    refresh(output)
    with pytest.raises(ValueError,match='Second baseline ran after unreliable first'):m.verify_output(output,DESIGN)


def test_direct_command_rows_preserve_full_wxyz_order_and_grip():
    rows=m.command_rows(REF['trajectory'])
    assert rows.shape==(246,16)
    for i in range(246):
        assert rows[i].tolist()==[*REF['trajectory']['left']['pose'][i],REF['trajectory']['left']['gripper'][i],*REF['trajectory']['right']['pose'][i],REF['trajectory']['right']['gripper'][i]]


def test_complete_replay_started_false_cannot_qualify_after_rehash(simulated):
    output,factory,_,_=simulated;m.execute_sequence(output,REF,DESIGN,factory)
    p=output/'trials/baseline_0/outcome.json';raw=m.read(p);raw['replay_started']=False;m.atomic_json(p,raw);refresh(output)
    with pytest.raises(ValueError,match='Complete replay requires replay_started'):m.verify_output(output,DESIGN)


def test_setup_primary_and_cleanup_errors_both_retained(simulated):
    output,_,_,_=simulated
    class Native:
        def close_env(self):raise RuntimeError('separate cleanup failure')
    def factory(seed):
        try:raise ValueError('primary setup failure')
        except BaseException as primary:
            m.preserve_setup_cleanup_error(Native(),primary)
            raise
    report=m.execute_sequence(output,REF,DESIGN,factory)
    assert report['native_initialization_attempts']==1 and report['baseline_replays']==0
    errors=report['rows'][0]['errors']
    assert errors[0]['type']=='ValueError'and errors[0]['message']=='primary setup failure'
    assert errors[1]['phase']=='setup_cleanup'and errors[1]['type']=='RuntimeError'and errors[1]['message']=='separate cleanup failure'
    assert 'native_initializations_started'not in report
